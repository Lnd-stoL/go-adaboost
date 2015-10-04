
package main

import (
	"fmt"
	"runtime"
	"flag"
    "os"
    "log"
    "runtime/pprof"

	adaboost "datamining-hw/adaboost_classifier"
	mlearn "datamining-hw/machine_learning"

    "github.com/gonum/plot"
    "github.com/gonum/plot/plotter"
    "github.com/gonum/plot/plotutil"
    "github.com/gonum/plot/vg"
)

//----------------------------------------------------------------------------------------------------------------------

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU() * 2)

	pMaxTreeHeight := flag.Int("tree_height", 30, "maximum height of base model trees")
	pNumBaseModels := flag.Int("num_estimators", 100, "maximum number of base models (estimator) used in adaboosting")
    memprofile     := flag.String("memprofile", "", "write memory profile to specified file")
    cpuprofile     := flag.String("cpuprofile", "", "write cpu profile to specified file")
    datasetName    := flag.String("dataset", "bupa", "selects dataset to operate on (predefined are: spam, iris, bupa, wines)")
    trainSubset    := flag.Float64("train_subset", 1.0, "percent of the dataset used to train on (the rest is used to test)")
    flag.Parse()

    // enabling CPU profiling
    if *cpuprofile != "" {
        f, err := os.Create(*cpuprofile)
        if err != nil {
            log.Fatal(err)
        }
        pprof.StartCPUProfile(f)
        defer pprof.StopCPUProfile()
    }

	fmt.Println("loading datasets ...")
	train_dataset, test_dataset := loadDatasets(*datasetName)
    if train_dataset == nil || test_dataset == nil {
        fmt.Println(os.Stderr, "can't load dataset " + *datasetName)
        return
    }

    // preparing train set
    train_dataset.SubsetRandomSamples(*trainSubset)
	train_dataset.GenerateArgOrderByFeatures()

    testAdaboostClassifier(*pNumBaseModels, *pMaxTreeHeight, train_dataset, test_dataset)

    //fmt.Println("performing CFS feature filtering ...")
    //filtered_features := train_dataset.SelectFeaturesWithCFS(6)
    //fmt.Println(filtered_features)
    //train_dataset.SubsetFeatures(filtered_features)
    //test_dataset.SubsetFeatures(filtered_features)

    //test_results, ref_f1 := testEmbeddedFeaturesFiltering(*pNumBaseModels, *pMaxTreeHeight, train_dataset, test_dataset)
    //drawEmbeddedFeaturesFilteringResults(test_results, ref_f1, "embedded_filtering.png")

    // enabling memory profiling
    writeMemProfile(memprofile)
}


func loadDatasets(datasetName string) (*mlearn.DataSet, *mlearn.DataSet){
    switch datasetName {
    case "spam":
        loaderParams := mlearn.DataSetLoaderParams{
            ClassesFirst: true,
            ClassesFromZero: true,
            Splitter: " ",
        }
        return mlearn.LoadDataSetFromFile("./datasets/spam/spam.train.txt", loaderParams),
               mlearn.LoadDataSetFromFile("./datasets/spam/spam.test.txt", loaderParams)

    case "iris":
        loaderParams := mlearn.DataSetLoaderParams{
            ClassesFirst: false,
            ClassesFromZero: false,
            Splitter: ",",
        }
        return mlearn.LoadDataSetFromFile("./datasets/iris.data", loaderParams),
               mlearn.LoadDataSetFromFile("./datasets/iris.data", loaderParams)

    case "bupa":
        loaderParams := mlearn.DataSetLoaderParams{
            ClassesFirst: false,
            ClassesFromZero: false,
            Splitter: ",",
        }
        return mlearn.LoadDataSetFromFile("./datasets/bupa.data", loaderParams),
               mlearn.LoadDataSetFromFile("./datasets/bupa.data", loaderParams)

    case "wine":
        loaderParams := mlearn.DataSetLoaderParams{
            ClassesFirst: true,
            ClassesFromZero: false,
            Splitter: ",",
        }
        return mlearn.LoadDataSetFromFile("./datasets/wine.data", loaderParams),
               mlearn.LoadDataSetFromFile("./datasets/wine.data", loaderParams)
    }

    return nil, nil
}


func writeMemProfile(memprofile *string) {
    if *memprofile == "" { return }

    f, err := os.Create(*memprofile)
    if err != nil {
        log.Fatal(err)
    }

    pprof.WriteHeapProfile(f)
    f.Close()
}


func testAdaboostClassifier(numBaseModels, maxTreeHeight int, train_dataset, test_dataset *mlearn.DataSet) {
    fmt.Println("training adaboost classifier over CART trees ...")
    cartTrainer := adaboost.NewCARTClassifierTrainer(train_dataset,
        adaboost.CARTClassifierTrainOptions{ MaxDepth: int(maxTreeHeight), MinElementsInLeaf: 10})

    adaboostTrainer := adaboost.NewAdaboostClassifierTrainer(cartTrainer)
    classifier := adaboostTrainer.TrainClassifier(train_dataset,
        adaboost.AdaboostClassifierTrainOptions{MaxEstimators: numBaseModels})

    fmt.Println("predicting ...")
    predictions := make([]int, len(test_dataset.Classes))
    for i := range predictions {
        predictions[i] = classifier.PredictProbe(test_dataset.GetSample(i))
    }

    precision, recall, f1 := mlearn.PrecisionRecallF1(predictions, test_dataset.Classes, test_dataset.ClassesNum)
    fmt.Printf("\nprecision: %.3v  recall: %.3v  f1: %.3v \n", precision, recall, f1)
}


type testEmbeddedFeaturesFilteringResult struct {
    FeaturesCount int
    F1 float64
}


func testEmbeddedFeaturesFiltering(numBaseModels, maxTreeHeight int, train_dataset, test_dataset *mlearn.DataSet) (
                                    []testEmbeddedFeaturesFilteringResult, float64) {

    fmt.Println("training adaboost classifier with embedded features filtering ...")
    cartTrainer := adaboost.NewCARTClassifierTrainer(train_dataset,
        adaboost.CARTClassifierTrainOptions{ MaxDepth: int(maxTreeHeight), MinElementsInLeaf: 10, EnableEmbeddedFeaturesRanking: true})

    adaboostTrainer := adaboost.NewAdaboostClassifierTrainer(cartTrainer)
    classifier := adaboostTrainer.TrainClassifier(train_dataset,
        adaboost.AdaboostClassifierTrainOptions{MaxEstimators: numBaseModels, EnableEmbeddedFeaturesRanking: true})
    ranked_features := adaboostTrainer.GetRankedFeatures()

    fmt.Println("predicting ...")
    predictions := make([]int, len(test_dataset.Classes))
    for i := range predictions {
        predictions[i] = classifier.PredictProbe(test_dataset.GetSample(i))
    }

    precision, recall, ref_f1 := mlearn.PrecisionRecallF1(predictions, test_dataset.Classes, test_dataset.ClassesNum)
    fmt.Printf("reference precision: %.3v  recall: %v  f1: %.3v \n\n", precision, recall, ref_f1)

    var test_results []testEmbeddedFeaturesFilteringResult
    for j := 10; j < test_dataset.FeaturesNum; j++ {
        fmt.Printf("now training again using only %v of selected features ...\n", j)
        selected_features := ranked_features[:j]
        fmt.Println("selected features are: ")
        fmt.Println(selected_features)

        train_data_subset := &mlearn.DataSet{}
        test_data_subset  := &mlearn.DataSet{}
        *train_data_subset = *train_dataset
        *test_data_subset  = *test_dataset
        train_data_subset.SubsetFeatures(selected_features)
        test_data_subset.SubsetFeatures(selected_features)

        fmt.Println()
        cartTrainer = adaboost.NewCARTClassifierTrainer(train_data_subset,
            adaboost.CARTClassifierTrainOptions{ MaxDepth: int(maxTreeHeight), MinElementsInLeaf: 10})

        adaboostTrainer = adaboost.NewAdaboostClassifierTrainer(cartTrainer)
        classifier = adaboostTrainer.TrainClassifier(train_data_subset,
            adaboost.AdaboostClassifierTrainOptions{MaxEstimators: numBaseModels})

        fmt.Println("predicting ...")
        predictions = make([]int, len(test_data_subset.Classes))
        for i := range predictions {
            predictions[i] = classifier.PredictProbe(test_data_subset.GetSample(i))
        }

        precision, recall, f1 := mlearn.PrecisionRecallF1(predictions, test_dataset.Classes, test_dataset.ClassesNum)
        fmt.Printf("precision: %.3v  recall: %.3v  f1: %.3v \n", precision, recall, f1)

        test_results = append(test_results, testEmbeddedFeaturesFilteringResult{FeaturesCount: j, F1: f1})
    }

    return test_results, ref_f1
}


func drawEmbeddedFeaturesFilteringResults(results []testEmbeddedFeaturesFilteringResult, refF1 float64, fileName string) {
    fmt.Println("plotting ...")

    p, err := plot.New()
    if err != nil { panic(err) }

    p.Title.Text = "Adaboost: Embedded features filtering test"
    p.X.Label.Text = "Selected features count"
    p.Y.Label.Text = "F1"

    var resultPoints plotter.XYs
    for _, nextResult := range results {
        resultPoints = append(resultPoints,
                struct { X, Y float64 } { float64(nextResult.FeaturesCount), nextResult.F1 })
    }

    refPoints := make(plotter.XYs, 2)
    refPoints[0] = struct { X, Y float64 } { float64(results[0].FeaturesCount), refF1 }
    refPoints[1] = struct { X, Y float64 } { float64(results[len(results)-1].FeaturesCount), refF1 }

    err = plotutil.AddLinePoints(p,
        "F1 with selected features", resultPoints,
        "reference F1 with all features", refPoints)
    if err != nil { panic(err) }

    if err := p.Save(12*vg.Inch, 12*vg.Inch, fileName); err != nil {
        panic(err)
    }
}
