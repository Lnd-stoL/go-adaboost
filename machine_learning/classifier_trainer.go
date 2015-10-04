
package machine_learning

//----------------------------------------------------------------------------------------------------------------------

type ClassifierTrainer interface {
    TrainClassifier(data_set *DataSet) BaseClassifier
    TrainClassifierWithWeights(data_set *DataSet, weights []float64) BaseClassifier

    GetRankedFeatures() []int
    GetFeaturesRank()   []float64
}

//----------------------------------------------------------------------------------------------------------------------
// just utility type for GetRankedFeatures()
//----------------------------------------------------------------------------------------------------------------------

type FeaturesRankSorter struct {
    FeaturesRank []float64
    Indices []int
}


func (by FeaturesRankSorter) Len() int {
    return len(by.Indices)
}

func (by FeaturesRankSorter) Swap(i, j int) {
    by.Indices[i], by.Indices[j] = by.Indices[j], by.Indices[i]

}

func (by FeaturesRankSorter) Less(i, j int) bool {
    return by.FeaturesRank[by.Indices[i]] > by.FeaturesRank[by.Indices[j]]
}
