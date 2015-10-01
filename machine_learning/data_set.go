
package machine_learning

import (
    "os"
    "bufio"
    "strings"
    "strconv"
    "sort"

    stat "github.com/gonum/stat"
    "math"
    "fmt"
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
        //cur_sample++

        //ds.Samples = append(ds.Samples, sample)
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


func (DataSet *DataSet) calculateCFS(features_correlation [][]float64, answer_correlation []float64,
                                     features_subset []int, features_subset_mask []bool) float64 {

    var d float64

    var ans_corr_sum float64
    for i, id := range features_subset {
        if !features_subset_mask[i] { continue }

        d += 1.0
        ans_corr_sum += math.Abs(answer_correlation[id])
    }

    var features_corr_sum float64
    for i, idi := range features_subset {
        for j, idj := range features_subset {
            if !features_subset_mask[j] || !features_subset_mask[i] { continue }

            features_corr_sum += math.Abs(features_correlation[idi][idj])
        }
    }

    return ans_corr_sum / math.Sqrt(d + features_corr_sum)
}


func copyIntSliceSubset(slice []int, mask []bool) []int {
    filtered_slice := make([]int, 0, len(slice))

    for i := range slice {
        if !mask[i] {
            continue
        }

        filtered_slice = append(filtered_slice, slice[i])
    }

    return filtered_slice
}


func (data_set *DataSet) FilterFeaturesWithCFS() []int {
    // prepare data
    features_correlation := make([][]float64, data_set.FeaturesNum)
    for i := range features_correlation {
        features_correlation[i] = make([]float64, data_set.FeaturesNum)
    }

    answer_correlation := make([]float64, data_set.FeaturesNum)
    answers := make([]float64, len(data_set.Classes))
    for i := range data_set.Classes {
        answers[i] = float64(data_set.Classes[i])
    }

    // calculate correlations
    for i := range data_set.SamplesByFeature {
        for j := 0; j < i; j++ {
            corr := stat.Correlation(data_set.SamplesByFeature[i], data_set.SamplesByFeature[j], nil)
            features_correlation[i][j] = corr
            features_correlation[j][i] = corr
        }
    }

    for i := range data_set.SamplesByFeature {
        answer_correlation[i] = stat.Correlation(data_set.SamplesByFeature[i], answers, nil)
    }

    // now maximize CFS
    maxCFS := 0.0
    var best_features_subset []int

    features_subset := make([]int, data_set.FeaturesNum)
    features_subset_mask := make([]bool, data_set.FeaturesNum)
    for i := range features_subset {
        features_subset[i] = i
        features_subset_mask[i] = true
    }

    for d := 0; d < 50; d++ {
        prev_i := 0
        best_i := 0
        var localMaxCFS float64

        for i := range features_subset {
            features_subset_mask[prev_i] = true
            features_subset_mask[i] = false
            prev_i = i

            cfs := data_set.calculateCFS(features_correlation, answer_correlation, features_subset, features_subset_mask)
            if cfs > localMaxCFS {
                localMaxCFS = cfs
                best_i = i
            }
        }
        features_subset_mask[prev_i] = true

        if localMaxCFS > maxCFS {
            fmt.Printf("%v: %v > %v\n", d, localMaxCFS, maxCFS)
            maxCFS = localMaxCFS
            features_subset_mask[best_i] = false
            best_features_subset = copyIntSliceSubset(features_subset, features_subset_mask)
        }

        features_subset = append(features_subset[:best_i], features_subset[best_i+1:]...)
        features_subset_mask = append(features_subset_mask[:best_i], features_subset_mask[best_i+1:]...)
    }

    fmt.Println(len(best_features_subset))
    return best_features_subset
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
