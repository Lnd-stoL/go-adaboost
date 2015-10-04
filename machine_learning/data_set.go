
package machine_learning

import (
    "os"
    "bufio"
    "strings"
    "strconv"
    "sort"
    "math"
    "math/rand"
    "time"

    stat "github.com/gonum/stat"
)

//----------------------------------------------------------------------------------------------------------------------

type DataSet struct {
    SamplesByFeature [][]float64
    //Samples [][]float64
    Classes []int

    ClassesNum  int
    FeaturesNum int
    SamplesNum  int

    ArgOrderedByFeature [][]int
}


type DataSetLoaderParams struct {
    ClassesFirst, ClassesFromZero bool
    Splitter string
}


func DataSetLoaderParams_Defaults() DataSetLoaderParams {
    return DataSetLoaderParams{
        ClassesFirst: true,
        ClassesFromZero: false,
        Splitter: ",",
    }
}


func LoadDataSetFromFile(fileName string, params DataSetLoaderParams) (ds *DataSet) {
    ds = new(DataSet)

    file, err := os.Open(fileName)
	if err != nil { panic(err) }
    defer file.Close()

    fileScanner := bufio.NewScanner(file)
    //cur_sample := 0
    for fileScanner.Scan() {
        numbers := strings.Split(fileScanner.Text(), params.Splitter)
        sample := make([]float64, len(numbers)-1)

        if ds.FeaturesNum == 0 {
            ds.FeaturesNum = len(sample)
            ds.SamplesByFeature = make([][]float64, ds.FeaturesNum)
        }

        j := 0
        var class_val int64
        for i, element := range numbers {
            if (params.ClassesFirst && i == 0) || (!params.ClassesFirst && i == len(numbers)-1) {
                class_val, _ = strconv.ParseInt(element, 10, 32)
                continue
            }

            sample[j], _ = strconv.ParseFloat(element, 64)
            j += 1
        }

        if !params.ClassesFromZero {
            class_val -= 1
        }

        if int(class_val) >= ds.ClassesNum {
            ds.ClassesNum = int(class_val) + 1
        }

        for i := range sample {
            ds.SamplesByFeature[i] = append(ds.SamplesByFeature[i], sample[i])
        }

        ds.Classes = append(ds.Classes, int(class_val))
    }

    ds.SamplesNum = len(ds.SamplesByFeature[0])
    return
}


func (ds *DataSet) GetSample(id int) []float64 {
    sample := make([]float64, ds.FeaturesNum)
    ds.GetSampleInplace(id, sample)
    return sample
}


func (ds *DataSet) GetSampleInplace(id int, sample []float64) {
    for i := range ds.SamplesByFeature {
        sample[i] = ds.SamplesByFeature[i][id]
    }
}


// sort helper used to sort samples by feature
type argSortByFeature struct {
    Samples []float64
    Indices []int
}

func (by argSortByFeature) Len() int {
    return len(by.Samples)
}

func (by argSortByFeature) Swap(i, j int) {
    by.Indices[i], by.Indices[j] = by.Indices[j], by.Indices[i]

}

func (by argSortByFeature) Less(i, j int) bool {
    return by.Samples[by.Indices[i]] < by.Samples[by.Indices[j]]
}


func (data_set *DataSet) GenerateArgOrderByFeatures() {
    arg_sorted_samples := make([][]int, data_set.FeaturesNum)
    straight_order_indices := make([]int, data_set.SamplesNum)

    for i := 0; i < data_set.SamplesNum; i++ {
        straight_order_indices[i] = i
    }

    for j := 0; j < data_set.FeaturesNum; j++ {
        indices := make([]int, len(straight_order_indices))
        copy(indices, straight_order_indices)
        sort.Sort(argSortByFeature{ Samples: data_set.SamplesByFeature[j], Indices: indices })
        arg_sorted_samples[j] = indices
    }

    data_set.ArgOrderedByFeature = arg_sorted_samples
}


func calculateCFS(features_correlation [][]float64, answer_correlation []float64, features_subset []int) float64 {
    var ans_corr_sum float64
    for _, id := range features_subset {
        ans_corr_sum += math.Abs(answer_correlation[id])
    }

    var features_corr_sum float64
    for _, idi := range features_subset {
        for _, idj := range features_subset {
            features_corr_sum += features_correlation[idi][idj]
        }
    }

    d := float64(len(features_subset))
    return ans_corr_sum / math.Sqrt(d + features_corr_sum)
}


func (data_set *DataSet) recursivelyMaximizeCSF(features_correlation [][]float64, answer_correlation []float64,
                                                features_subset []int, features_count int, bestCSF float64) (float64, []int) {
    if len(features_subset) == features_count {
        return calculateCFS(features_correlation, answer_correlation, features_subset), features_subset
    }

    var best_features_subset []int
    for i := features_subset[features_count-1]+1; i < data_set.FeaturesNum; i++ {
        features_subset[features_count] = i

        csf, next_features_subset :=
            data_set.recursivelyMaximizeCSF(features_correlation, answer_correlation, features_subset, features_count+1, bestCSF)

        if csf > bestCSF {
            bestCSF = csf
            if best_features_subset == nil {
                best_features_subset = make([]int, len(features_subset))
            }
            copy(best_features_subset, next_features_subset)
        }
    }

    return bestCSF, best_features_subset
}


func (data_set *DataSet) calculateCorrelations() (features_correlation [][]float64, answer_correlation []float64) {
    // prepare data
    features_correlation = make([][]float64, data_set.FeaturesNum)
    for i := range features_correlation {
        features_correlation[i] = make([]float64, data_set.FeaturesNum)
    }

    answer_correlation = make([]float64, data_set.FeaturesNum)
    answers := make([]float64, len(data_set.Classes))
    for i := range data_set.Classes {
        answers[i] = float64(data_set.Classes[i])
    }

    // calculate correlations
    for i := range data_set.SamplesByFeature {
        for j := 0; j < i; j++ {
            corr := math.Abs(stat.Correlation(data_set.SamplesByFeature[i], data_set.SamplesByFeature[j], nil))
            features_correlation[i][j] = corr
            features_correlation[j][i] = corr
        }
    }

    for i := range data_set.SamplesByFeature {
        answer_correlation[i] = math.Abs(stat.Correlation(data_set.SamplesByFeature[i], answers, nil))
    }

    return
}


func (data_set *DataSet) SelectFeaturesWithCFS(features_count int) []int {
    features_correlation, answer_correlation := data_set.calculateCorrelations()

    features_subset := make([]int, features_count)
    for i := range features_subset {
        features_subset[i] = i
    }

    _, best_features_subset := data_set.recursivelyMaximizeCSF(features_correlation, answer_correlation, features_subset, 1, 0)
    return best_features_subset
}


func (data_set *DataSet) SelectFeaturesWithDispersion(dispersion float64) []int {
    return make([]int, 1)
}


func (data_set *DataSet) SubsetFeatures(features_subset []int) {
    newSamplesByFeatures := make([][]float64, len(features_subset))
    for i, id := range features_subset {
        newSamplesByFeatures[i] = data_set.SamplesByFeature[id]
    }
    data_set.SamplesByFeature = newSamplesByFeatures

    if data_set.ArgOrderedByFeature != nil {
        newArgOrderedByFeature := make([][]int, len(features_subset))
        for i, id := range features_subset {
            newArgOrderedByFeature[i] = data_set.ArgOrderedByFeature[id]
        }
        data_set.ArgOrderedByFeature = newArgOrderedByFeature
    }

    data_set.FeaturesNum = len(features_subset)
}


func randomShuffleInts(a []int) {
    for i := range a {
        j := rand.Intn(i + 1)
        a[i], a[j] = a[j], a[i]
    }
}


func subsetFloats(data []float64, indices []int) []float64 {
    subset := make([]float64, len(indices))

    for i, id := range indices {
        subset[i] = data[id]
    }

    return subset
}


func subsetInts(data []int, indices []int) []int {
    subset := make([]int, len(indices))

    for i, id := range indices {
        subset[i] = data[id]
    }

    return subset
}


func (data_set *DataSet) SubsetRandomSamples(percent float64) {
    if percent >= 1.0 { return }

    if data_set.ArgOrderedByFeature != nil {
        panic("Not implemented")
    }

    indices := make([]int, data_set.SamplesNum)
    for i := range indices {
        indices[i] = i
    }

    rand.Seed(time.Now().UnixNano())
    randomShuffleInts(indices)
    indices_subset := indices[:int(float64(len(indices)) * percent)]

    for j := range data_set.SamplesByFeature {
        data_set.SamplesByFeature[j] = subsetFloats(data_set.SamplesByFeature[j], indices_subset)
    }
    data_set.Classes = subsetInts(data_set.Classes, indices_subset)

    data_set.SamplesNum = len(indices_subset)
}
