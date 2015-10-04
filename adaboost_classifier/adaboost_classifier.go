
package adaboost_classifier

import (
    "fmt"
    "math"
    "sort"

    mlearn "datamining-hw/machine_learning"
)

//----------------------------------------------------------------------------------------------------------------------


type AdaboostClassifier struct {
    base_estimators  []mlearn.BaseClassifier
    combination_coeffs []float64
    classes_num int
}


type AdaboostClassifierTrainOptions struct {
    MaxEstimators int
    TargetError float64
    EnableEmbeddedFeaturesRanking bool
}


type AdaboostClassifierTrainer struct {
    prediction []int
    weights    []float64

    baseModelTrainer mlearn.ClassifierTrainer
    options AdaboostClassifierTrainOptions

    EmbeddedFeaturesRank []float64
}


func NewAdaboostClassifierTrainer(baseEstimatorTrainer mlearn.ClassifierTrainer) *AdaboostClassifierTrainer {
    trainer := &AdaboostClassifierTrainer{baseModelTrainer: baseEstimatorTrainer}
    return trainer
}


func (trainer *AdaboostClassifierTrainer) TrainClassifier(data_set *mlearn.DataSet,
                                                          options AdaboostClassifierTrainOptions) *AdaboostClassifier {

    classifier := &AdaboostClassifier{classes_num: data_set.ClassesNum}
    trainer.options = options
    trainer.prediction = make([]int, data_set.SamplesNum)
    trainer.weights = make([]float64, data_set.SamplesNum)

    if trainer.options.EnableEmbeddedFeaturesRanking {
        trainer.EmbeddedFeaturesRank = make([]float64, data_set.FeaturesNum)
    }

    // generate initial equal normalized weights
    for i := range trainer.weights {
		trainer.weights[i] = 1.0 / float64(data_set.SamplesNum)
	}

    var err_sum, err_count float64
    for i := 0; i < options.MaxEstimators; i++ {
        b, estimator, err := trainer.trainNextBaseEstimator(data_set, i)
        classifier.base_estimators = append(classifier.base_estimators, estimator)
        classifier.combination_coeffs = append(classifier.combination_coeffs, b)

        if  err <= options.TargetError {
            fmt.Printf("adaboost: boosting stopped with %v estimators; err = %.4v\n", len(classifier.base_estimators), err)
            break
        } else {
            err_sum += err
            err_count += 1.0

            if (i+1)%10 == 0 || i == options.MaxEstimators-1 {
                fmt.Printf("adaboost: trained base estimator #%v with self err = %.4v\n", len(classifier.base_estimators), err_sum / err_count)
                err_sum = 0
                err_count = 0
            }
        }
    }

    return classifier
}


func (ac *AdaboostClassifier) PredictProbe(probe []float64) int {
    var max_result float64
    var max_result_class int

    for k := 0; k < ac.classes_num; k++ {
        var result float64
        for i := range ac.base_estimators {
            if ac.base_estimators[i].PredictProbe(probe) == k {
                result += ac.combination_coeffs[i]
            }
        }

        if result > max_result {
            max_result = result
            max_result_class = k
        }
    }

    return max_result_class
}


func (trainer *AdaboostClassifierTrainer) trainNextBaseEstimator(data_set *mlearn.DataSet, step int) (
                                                                 float64, mlearn.BaseClassifier, float64) {
    base_estimator := trainer.baseModelTrainer.TrainClassifierWithWeights(data_set, trainer.weights)
    // embedded features ranking
    if (trainer.options.EnableEmbeddedFeaturesRanking) {
        for j := range trainer.EmbeddedFeaturesRank {
            trainer.EmbeddedFeaturesRank[j] += trainer.baseModelTrainer.GetFeaturesRank()[j]
        }
    }

    // save the predictions for later use
    sample := make([]float64, data_set.FeaturesNum)
    for i := range trainer.prediction {
        data_set.GetSampleInplace(i, sample)
        trainer.prediction[i] = base_estimator.PredictProbe(sample)
    }

    // calculate the weighted prediction error
    var err float64
    for i := 0; i < data_set.SamplesNum; i++ {
        if trainer.prediction[i] != data_set.Classes[i] {
            err += trainer.weights[i]
        }
    }

    // check it to prevent math.Log domain violation (err == 0 means that there's nothing to boost)
    if err == 0 {
        return 1.0, base_estimator, err
    }
    b := math.Log((1.0 - err) / err) + math.Log(float64(data_set.ClassesNum) - 1.0)

    // update weights according to error
    const minimalWeight float64 = 10e-9
    var weights_sum float64
    for i := range trainer.weights {
        if trainer.prediction[i] != data_set.Classes[i] {
            trainer.weights[i] *= math.Exp(b)

            if trainer.weights[i] < minimalWeight {   /* this is to deal with rounding errors */
                trainer.weights[i] = 0
            }
        } else {
            //trainer.weights[i] *= math.Exp(-b)
        }
        weights_sum += trainer.weights[i]
    }

    // normalize weights to 1.0 sum
    var normalized_sum float64
    for i := range trainer.weights {
        trainer.weights[i] /= weights_sum
        normalized_sum += trainer.weights[i]
    }
    // dirty hack to guarantee 1.0 weights sum which is important to base model (regardless of rounding errors)
    trainer.weights[0] += 1.0 - normalized_sum

    return b, base_estimator, err
}


func (trainer *AdaboostClassifierTrainer) GetRankedFeatures() []int {
    indices := make([]int, len(trainer.EmbeddedFeaturesRank))
    for i := range indices {
        indices[i] = i
    }

    sort.Sort(mlearn.FeaturesRankSorter{FeaturesRank: trainer.EmbeddedFeaturesRank, Indices: indices})
    return indices
}


func (trainer *AdaboostClassifierTrainer) GetFeaturesRank() []float64 {
    return trainer.EmbeddedFeaturesRank
}
