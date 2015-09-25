
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
)

//----------------------------------------------------------------------------------------------------------------------

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU() * 2)

	pMaxTreeHeight := flag.Int("tree_height", 3, "maximum height of base model trees")
	pNumBaseModels := flag.Int("num_estimators", 100, "maximum number of base models (estimator) used in adaboosting")
    memprofile     := flag.String("memprofile", "", "write memory profile to this file")
    cpuprofile     := flag.String("cpuprofile", "", "write cpu profile to file")
	flag.Parse()

    // enabling profiling
    if *cpuprofile != "" {
        f, err := os.Create(*cpuprofile)
        if err != nil {
            log.Fatal(err)
        }
        pprof.StartCPUProfile(f)
        defer pprof.StopCPUProfile()
    }

	fmt.Println("loading datasets ...")
	train_dataset, test_dataset := loadTrainDataset(), loadTestDataset()
	train_dataset.GenerateArgOrderByFeatures()

	fmt.Println("training classifier ...")
    cartTrainer := adaboost.NewCARTClassifierTrainer(train_dataset)
    baseModelTrainer := func(dataSet *mlearn.DataSet, weights []float64, step int) mlearn.BaseEstimator {
        maxDepth := *pMaxTreeHeight
        return cartTrainer.TrainClassifier(dataSet, weights,
            adaboost.CARTClassifierTrainOptions{ MaxDepth: int(maxDepth), MinElementsInLeaf: 10})
    }

	classifier := adaboost.TrainAdaboostClassifier(train_dataset, baseModelTrainer,
		adaboost.AdaboostClassifierTrainOptions{MaxEstimators: *pNumBaseModels})

    // enabling memory profiling
    writeMemProfile(memprofile)

    fmt.Println("predicting ...")
	predictions := make([]int, len(test_dataset.Classes))
	for i := range predictions {
		predictions[i] = classifier.PredictProbe(test_dataset.GetSample(i))
	}

	precision, recall, f1 := mlearn.PrecisionRecallF1(predictions, test_dataset.Classes, test_dataset.ClassesNum)
	fmt.Printf("\nprecision: %v  recall: %v  f1: %v \n", precision, recall, f1)
}


func loadTrainDataset() *mlearn.DataSet {
	return mlearn.LoadDataSetFromFile("./datasets/spam/spam.train.txt", mlearn.DataSetLoaderParams {
        ClassesFirst: true,
        ClassesFromZero: true,
        Splitter: " ",
    })
}


func loadTestDataset() *mlearn.DataSet {
	return mlearn.LoadDataSetFromFile("./datasets/spam/spam.test.txt", mlearn.DataSetLoaderParams {
        ClassesFirst: true,
        ClassesFromZero: true,
        Splitter: " ",
    })
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
