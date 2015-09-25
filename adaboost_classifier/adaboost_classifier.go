
package adaboost_classifier

import (
    "fmt"
    "math"
    mlearn "datamining-hw/machine_learning"
)

//----------------------------------------------------------------------------------------------------------------------

type BaseEstimatorTrainerDelegate func(
    dataSet *mlearn.DataSet, weights []float64, step int) mlearn.BaseEstimator


type AdaboostClassifier struct {
    base_estimators  []mlearn.BaseEstimator
    combination_coeffs []float64
    classes_num int
}


type AdaboostClassifierTrainOptions struct {
    MaxEstimators int
    TargetError float64
}


type adaboostClassifierTrainer struct {
    prediction []int
    weights    []float64

    baseModelTrainer BaseEstimatorTrainerDelegate
    options AdaboostClassifierTrainOptions
}


func TrainAdaboostClassifier(data_set *mlearn.DataSet,
                             baseEstimatorTrainer BaseEstimatorTrainerDelegate,
                             options AdaboostClassifierTrainOptions) *AdaboostClassifier {

    classifier := &AdaboostClassifier{classes_num: data_set.ClassesNum}
    trainer := adaboostClassifierTrainer{options: options, baseModelTrainer: baseEstimatorTrainer}
    trainer.prediction = make([]int, data_set.SamplesNum)
    trainer.weights = make([]float64, data_set.SamplesNum)

    // generate initial equal normalized weights
    for i := range trainer.weights {
		trainer.weights[i] = 1.0 / float64(data_set.SamplesNum)
	}

    for i := 0; i < options.MaxEstimators; i++ {
        if b, estimator, err := trainer.trainNextBaseEstimator(data_set, i); err <= options.TargetError {
            break
        } else {
            classifier.base_estimators = append(classifier.base_estimators, estimator)
            classifier.combination_coeffs = append(classifier.combination_coeffs, b)

            if (i+1)%10 == 0 || i == options.MaxEstimators-1 {
                fmt.Printf("adaboost: trained base estimator #%v with self err = %v\n", len(classifier.base_estimators), err)
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


func (trainer *adaboostClassifierTrainer) trainNextBaseEstimator(data_set *mlearn.DataSet, step int) (float64, mlearn.BaseEstimator, float64) {
    base_estimator := trainer.baseModelTrainer(data_set, trainer.weights, step)

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

    // check it to prevent math.Log domain violation
    if err == 0 {
        return 1.0, base_estimator, err
    }
    b := math.Log((1.0 - err) / err) + math.Log(float64(data_set.ClassesNum) - 1.0)

    // update weights according to error
    const minimalWeight float64 = 10e-7
    var weights_sum float64
    for i := range trainer.weights {
        if trainer.prediction[i] != data_set.Classes[i] {
            trainer.weights[i] *= math.Exp(b)

            if trainer.weights[i] < minimalWeight {   /* this is to deal with roudning errors */
                trainer.weights[i] = 0
            }
        } else {
            //weights[i] *= math.Exp(-b)
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
