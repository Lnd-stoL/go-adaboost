
package machine_learning

import (
    "os"
    "bufio"
    "strings"
    "strconv"
    "sort"
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
