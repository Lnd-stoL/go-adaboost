
package adaboost_classifier

import (
    "fmt"
    "sync"
    "sort"

    mlearn "datamining-hw/machine_learning"
)

//----------------------------------------------------------------------------------------------------------------------

type CARTClassifier struct {
    tree_root *treeNode
}


type CARTClassifierTrainOptions struct {
    MaxDepth int
    TargetImpurity float64
    MinElementsInLeaf int
    EnableEmbeddedFeaturesRanking bool
}


type CARTClassifierTrainer struct {
    weights []float64

    sampleIndicesPool sync.Pool
    classesDistPool   sync.Pool

    Options CARTClassifierTrainOptions
    EmbeddedFeaturesRank []float64
}


// internal helper structures
type (
    treeNode struct {
        split_value float64
        split_feature int

        left_child, right_child *treeNode
        left_class, right_class  int          /* if one of them == -1 than left_child are not null */
    }


    splitPartInfo struct {
        class int
        impurity float64
        weight float64
        samplesCount int
        classes_dist []float64
    }


    splitPartsInfo struct {
        left_part  splitPartInfo
        right_part splitPartInfo
    }


    testSplitInfo struct {
        impurity float64
        split_pos int
        split_parts splitPartsInfo
        feature_id int
    }


    argSortedSamplesBuffered struct {
        indices [][]int
        refs int
    }
)


func (cc *CARTClassifier) PredictProbe(probe []float64) int {
    return cc.predictProbe(probe, cc.tree_root)
}


func (cc *CARTClassifier) predictProbe(probe []float64, node *treeNode) int {
    if probe[node.split_feature] <= node.split_value {
        if node.left_class == -1 {
            return cc.predictProbe(probe, node.left_child)
        } else {
            return node.left_class
        }

    } else {

        if node.right_class == -1 {
            return cc.predictProbe(probe, node.right_child)
        } else {
            return node.right_class
        }
    }
}


func NewCARTClassifierTrainer(data_set *mlearn.DataSet, options CARTClassifierTrainOptions) *CARTClassifierTrainer {
    trainer := CARTClassifierTrainer{Options: options}

    trainer.sampleIndicesPool.New = func() interface{} {
        buffer := make([]int, data_set.SamplesNum * data_set.FeaturesNum)
        argSortedSamples := make([][]int, data_set.FeaturesNum*2)

        for i := range argSortedSamples {
            j := i % data_set.FeaturesNum
            argSortedSamples[i] = buffer[j*data_set.SamplesNum : (j+1)*data_set.SamplesNum]
        }
        return argSortedSamples
    }

    trainer.classesDistPool.New = func() interface{} {
        return make([]float64, data_set.ClassesNum)
    }

    return &trainer
}


func (trainer *CARTClassifierTrainer) TrainClassifierWithWeights(data_set *mlearn.DataSet, weights []float64) mlearn.BaseClassifier {
    classifier := &CARTClassifier{}
    trainer.weights = weights

    // zero target impurity is not tolerant to float rounding errors
    const minTargetImpurity = 10e-9
    if trainer.Options.TargetImpurity < minTargetImpurity {
        trainer.Options.TargetImpurity = minTargetImpurity
    }

    // reasonable default
    const defaultMinimalElementsInLeaf = 3
    if trainer.Options.MinElementsInLeaf == 0 {
        trainer.Options.MinElementsInLeaf = defaultMinimalElementsInLeaf
    }

    if trainer.Options.EnableEmbeddedFeaturesRanking {
        trainer.EmbeddedFeaturesRank = make([]float64, data_set.FeaturesNum)
    }

    classifier.tree_root = trainer.buildTree(data_set)
    return classifier
}


func (trainer *CARTClassifierTrainer) TrainClassifier(data_set *mlearn.DataSet) mlearn.BaseClassifier {
    ident_weights := make([]float64, data_set.SamplesNum)
    for i := range ident_weights {
        ident_weights[i] = 1.0 / float64(data_set.SamplesNum)
    }

    return trainer.TrainClassifierWithWeights(data_set, ident_weights)
}


func (trainer *CARTClassifierTrainer) buildTree(data_set *mlearn.DataSet) *treeNode {
    classes_dist := trainer.classesDistPool.Get().([]float64)
    for i := range classes_dist { classes_dist[i] = 0 }

    for i, c := range data_set.Classes {
        classes_dist[c] += trainer.weights[i]
    }

    arg_ordered_samples_buffer := argSortedSamplesBuffered {
        refs: 2,
        indices: data_set.ArgOrderedByFeature,
    }
    return trainer.makeNode(data_set, &arg_ordered_samples_buffer, classes_dist, 1.0, trainer.Options.MaxDepth)
}


func makeTerminalNode(split_feature int, split_value float64, left_class int, right_class int) *treeNode {
    return &treeNode{ split_value: split_value,
                      split_feature: split_feature,
                      left_class: left_class,
                      right_class: right_class }
}


func (tr *CARTClassifierTrainer) makeNode(data_set *mlearn.DataSet, arg_sorted_samples *argSortedSamplesBuffered,
                                          classes_dist []float64, sum_weight float64, depth int) *treeNode {

    split_feature, split_val, parts_info := tr.findBestSplit(data_set, arg_sorted_samples.indices, classes_dist, sum_weight)

    // maximum tree height exceeded
    if depth == 1 {
        return makeTerminalNode(split_feature, split_val, parts_info.left_part.class, parts_info.right_part.class)
    }

    node := new(treeNode)
    node.split_feature = split_feature
    node.split_value = split_val
    node.left_class = -1
    node.right_class = -1

    new_arg_sorted_samples_buffer := tr.sampleIndicesPool.Get().([][]int)
    new_arg_sorted_samples_left, new_arg_sorted_samples_right :=
        tr.splitSamples(data_set, arg_sorted_samples.indices, new_arg_sorted_samples_buffer, split_val,
                         split_feature, parts_info.left_part.samplesCount-1)

    arg_sorted_samples.refs--
    if arg_sorted_samples.refs <= 0 {
        tr.sampleIndicesPool.Put(arg_sorted_samples.indices)
    }

    const minimumSamplesNumForParallel = 256
    barrierChannel := make(chan int)
    var parallelRecursiveCalls int

    // the node is already pure, no need to split it recursively
    if parts_info.left_part.impurity <= tr.Options.TargetImpurity || parts_info.left_part.samplesCount < tr.Options.MinElementsInLeaf {
        node.left_class = parts_info.left_part.class
    } else {
        new_arg_sorted_samples := argSortedSamplesBuffered{ indices: new_arg_sorted_samples_left, refs: 2 }
        if len(new_arg_sorted_samples.indices[0]) < minimumSamplesNumForParallel {
            node.left_child = tr.makeNode(data_set, &new_arg_sorted_samples, parts_info.left_part.classes_dist, parts_info.left_part.weight, depth - 1)
        } else {
            parallelRecursiveCalls++
            go func(new_arg_sorted_samples *argSortedSamplesBuffered) {
                node.left_child = tr.makeNode(data_set, new_arg_sorted_samples, parts_info.left_part.classes_dist, parts_info.left_part.weight, depth - 1)
                barrierChannel <- 1
            }(&new_arg_sorted_samples)
        }
    }

    if parts_info.right_part.impurity <= tr.Options.TargetImpurity || parts_info.right_part.samplesCount < tr.Options.MinElementsInLeaf {
        node.right_class = parts_info.right_part.class
    } else {
        new_arg_sorted_samples := argSortedSamplesBuffered{ indices: new_arg_sorted_samples_right, refs: 2 }
        if len(new_arg_sorted_samples.indices[0]) < minimumSamplesNumForParallel {
            node.right_child = tr.makeNode(data_set, &new_arg_sorted_samples, parts_info.right_part.classes_dist, parts_info.right_part.weight, depth - 1)
        } else {
            parallelRecursiveCalls++
            go func(new_arg_sorted_samples *argSortedSamplesBuffered) {
                node.right_child = tr.makeNode(data_set, new_arg_sorted_samples, parts_info.right_part.classes_dist, parts_info.right_part.weight, depth - 1)
                barrierChannel <- 2
            }(&new_arg_sorted_samples)
        }
    }

    // waiting for all recursive calls to return here
    for parallelRecursiveCalls > 0 {
        <-barrierChannel
        parallelRecursiveCalls--
    }

    return node
}


func (tr *CARTClassifierTrainer) splitSamples(data_set *mlearn.DataSet, arg_sorted_samples [][]int, filtered_samples_buffer [][]int,
                                              filter_val float64, split_feature, split_pos int) ([][]int, [][]int) {
    filtered_samples_left  := filtered_samples_buffer
    filtered_samples_right := filtered_samples_left[data_set.FeaturesNum:]
    filtered_samples_left   = filtered_samples_left[:data_set.FeaturesNum]

    for j := 0; j < data_set.FeaturesNum; j++ {
        filtered_left_samples_feature  := filtered_samples_buffer[j][:split_pos+1]
        filtered_right_samples_feature := filtered_samples_buffer[j][split_pos+1:len(arg_sorted_samples[0])]

        filtered_left_samples_count  := 0
        filtered_right_samples_count := 0

        for _, i := range arg_sorted_samples[j] {
            if data_set.SamplesByFeature[split_feature][i] < filter_val {
                if filtered_left_samples_count >= len(filtered_left_samples_feature) {
                    filtered_left_samples_count++
                    continue
                }
                filtered_left_samples_feature[filtered_left_samples_count] = i
                filtered_left_samples_count++
            } else {
                filtered_right_samples_feature[filtered_right_samples_count] = i
                filtered_right_samples_count++
            }
        }

        filtered_samples_left[j]  = filtered_left_samples_feature
        filtered_samples_right[j] = filtered_right_samples_feature
    }

    return filtered_samples_left, filtered_samples_right
}



func (tr *CARTClassifierTrainer) findBestSplitByFeaturesChunk(data_set *mlearn.DataSet, arg_sorted_samples [][]int, features_subset [2]int,
                                                               classes_dist []float64, sum_weight float64) testSplitInfo {
    var best_split_info testSplitInfo
    best_split_info.impurity   = 1.0
    best_split_info.feature_id = -1
    best_split_info.split_pos  = -3

    for j := features_subset[0]; j < features_subset[1]; j++ {
        impurity, best_split_pos, best_split_parts :=
            tr.findBestSplitPosition(data_set, arg_sorted_samples[j], j, classes_dist, sum_weight)

        if impurity < best_split_info.impurity {
            best_split_info = testSplitInfo{ impurity: impurity, split_pos: best_split_pos,
                                             split_parts: best_split_parts, feature_id: j }
        }
    }

    return best_split_info
}


func minInt(x, y int) int {
    if x < y {
        return x
    }

    return y
}


func (tr *CARTClassifierTrainer) findBestSplit(data_set *mlearn.DataSet, arg_sorted_samples [][]int, classes_dist []float64, sum_weight float64) (
                                               best_split_feature int, split_val float64, parts_info splitPartsInfo) {
    const parallelTreadsCount = 4
    const minimumSamplesForParallel = 2048

    best_split_feature = -2
    best_impurity := 1.0
    split_pos := -1

    // really no performance profit with parallel threads
    if len(arg_sorted_samples[0]) < minimumSamplesForParallel {
        bounds := [2]int{ 0, data_set.FeaturesNum }
        next_split_test := tr.findBestSplitByFeaturesChunk(data_set, arg_sorted_samples, bounds, classes_dist, sum_weight)

        best_impurity      = next_split_test.impurity
        best_split_feature = next_split_test.feature_id
        split_pos          = next_split_test.split_pos
        parts_info         = next_split_test.split_parts

    } else {

    // maybe some speed up with parallel processing
        channel := make(chan testSplitInfo, parallelTreadsCount)
        parallel_parts := parallelTreadsCount
        if data_set.FeaturesNum % parallelTreadsCount != 0 {
            parallel_parts++
        }
        features_for_one_thread := data_set.FeaturesNum / parallelTreadsCount

        routine := func(j int) {
            bounds := [2]int{ features_for_one_thread * j, minInt(features_for_one_thread * (j + 1), data_set.FeaturesNum) }
            channel <- tr.findBestSplitByFeaturesChunk(data_set, arg_sorted_samples, bounds, classes_dist, sum_weight)
        }

        // map features subsets to workers
        for j := 0; j < parallel_parts; j++ {
            go routine(j)
        }

        // reduce to the best split
        for j := 0; j < parallel_parts; j++ {
            next_split_test := <- channel

            if next_split_test.impurity < best_impurity ||
            /* this magic is needed to make results channel-ordering-independent */
                (next_split_test.impurity == best_impurity && next_split_test.feature_id < best_split_feature) {

                best_impurity      = next_split_test.impurity
                best_split_feature = next_split_test.feature_id
                split_pos          = next_split_test.split_pos
                parts_info         = next_split_test.split_parts
            }
        }
    }

    split_left_id  := arg_sorted_samples[best_split_feature][split_pos]
    split_right_id := arg_sorted_samples[best_split_feature][split_pos + 1]

    split_val = (data_set.SamplesByFeature[best_split_feature][split_left_id] +
                 data_set.SamplesByFeature[best_split_feature][split_right_id]) / 2.0

    parts_info.left_part.samplesCount  = split_pos + 1
    parts_info.right_part.samplesCount = len(arg_sorted_samples[best_split_feature]) - split_pos - 1

    // embedded features ranking
    if tr.Options.EnableEmbeddedFeaturesRanking {
        impurity_improvement := giniImpurity(classes_dist, sum_weight) - best_impurity
        tr.EmbeddedFeaturesRank[best_split_feature] += impurity_improvement
    }

    tr.classesDistPool.Put(classes_dist)
    return
}


func argmax(slice []float64) int {
    // special two-class case handling to improve performance
    if len(slice) == 2 {
        if slice[0] > slice[1] {
            return 0
        } else {
            return 1
        }
    }

    max_i := 0
    max_v := slice[0]

    for i, v := range slice {
        if v > max_v {
            max_v = v
            max_i = i
        }
    }

    return max_i
}


func (tr *CARTClassifierTrainer) findBestSplitPosition(data_set *mlearn.DataSet, cur_subset_indices []int,
                                                        cur_feature int,
                                                        classes_dist []float64,
                                                        sum_weight float64) (best_split_impurity float64,
                                                                             best_split_pos int,
                                                                             best_split_parts splitPartsInfo) {
    best_split_impurity = 1.0
    best_split_pos = -1

    // initialize left classes distribution
    left_classes_dist := tr.classesDistPool.Get().([]float64)
    for i := range left_classes_dist { left_classes_dist[i] = 0 }
    left_part_weight  := float64(0.0)

    // initialize right classes distribution
    right_classes_dist := tr.classesDistPool.Get().([]float64)
    copy(right_classes_dist, classes_dist)
    right_part_weight := sum_weight

    prev_feature_val := data_set.SamplesByFeature[cur_feature][cur_subset_indices[0]]
    for i := 0; i < len(cur_subset_indices)-1; i++ {
        sample_index := cur_subset_indices[i]
        next_sample_index := cur_subset_indices[i+1]

        c := data_set.Classes[sample_index]
        w := tr.weights[sample_index]

        left_classes_dist[c]  += w
        left_part_weight      += w

        right_classes_dist[c] -= w
        right_part_weight     -= w

        next_feature_val := data_set.SamplesByFeature[cur_feature][next_sample_index]
        if prev_feature_val == next_feature_val {
            continue
        }
        prev_feature_val = next_feature_val

        left_impurity  := giniImpurity(left_classes_dist, left_part_weight)
        right_impurity := giniImpurity(right_classes_dist, right_part_weight)
        split_impurity := (left_part_weight * left_impurity + right_part_weight * right_impurity) / sum_weight

        left_part_class  := argmax(left_classes_dist)
        right_part_class := argmax(right_classes_dist)

        if split_impurity <= best_split_impurity {
            best_split_impurity = split_impurity
            best_split_pos = i

            if best_split_parts.left_part.classes_dist == nil {
                best_split_parts.left_part.classes_dist  = tr.classesDistPool.Get().([]float64)
                best_split_parts.right_part.classes_dist = tr.classesDistPool.Get().([]float64)
            }

            left_classes_dist_old, right_classes_dist_old := best_split_parts.left_part.classes_dist,
                                                             best_split_parts.right_part.classes_dist

            best_split_parts.left_part  = splitPartInfo{ class:  left_part_class,   impurity:     left_impurity,
                                                         weight: left_part_weight,  classes_dist: left_classes_dist  }

            best_split_parts.right_part = splitPartInfo{ class:  right_part_class,  impurity:     right_impurity,
                                                         weight: right_part_weight, classes_dist: right_classes_dist }

            left_classes_dist, right_classes_dist = left_classes_dist_old, right_classes_dist_old
            copy(left_classes_dist,  best_split_parts.left_part.classes_dist)
            copy(right_classes_dist, best_split_parts.right_part.classes_dist)
        }
    }

    tr.classesDistPool.Put(left_classes_dist)
    tr.classesDistPool.Put(right_classes_dist)

    return
}


func giniImpurity(classes_portions []float64, sum_weight float64) float64 {
    var sqSum float64
    for i := range classes_portions {
        sqSum += classes_portions[i] * classes_portions[i]
    }

    return 1.0 - sqSum / (sum_weight * sum_weight)
}


func (tr *CARTClassifierTrainer) GetRankedFeatures() []int {
    indices := make([]int, len(tr.EmbeddedFeaturesRank))
    for i := range indices {
        indices[i] = i
    }

    sort.Sort(mlearn.FeaturesRankSorter{FeaturesRank: tr.EmbeddedFeaturesRank, Indices: indices})
    return indices
}


func (tr *CARTClassifierTrainer) GetFeaturesRank() []float64 {
    return tr.EmbeddedFeaturesRank
}


func (cc *CARTClassifier) Dump() {
    cc.dump(cc.tree_root)
}


func (cc *CARTClassifier) dump(node *treeNode) {
    fmt.Println(*node)

    if node.left_class == -1 {
        cc.dump(node.left_child)
    }
    if node.right_class == -1 {
        cc.dump(node.right_child)
    }
}


func (cc *CARTClassifier) CloneEmpty() mlearn.BaseClassifier {
    clone := new(CARTClassifier)
    *clone = *cc
    clone.tree_root = nil

    return clone
}
