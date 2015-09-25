
package machine_learning

//----------------------------------------------------------------------------------------------------------------------

func PrecisionRecallF1(prediction []int, classes []int, num_classes int) (float64, float64, float64){
    if len(prediction) != len(classes) {
        panic("prediction and classes are not equal length")
    }

    type classPrediction struct{fp, fn, tp, tn, support int; recall, precision, f1 float64}
    class_predictions := make([]classPrediction, num_classes)

    // firstly calc fp, fn, tp, tn for each class
    for j := 0; j < num_classes; j++ {
        for i := range prediction {
            if classes[i] == j {
                class_predictions[j].support++
            }

            switch  {
            case classes[i] != j && prediction[i] != j:
                class_predictions[j].tn++

            case classes[i] != j && prediction[i] == j:
                class_predictions[j].fp++

            case classes[i] == j && prediction[i] == j:
                class_predictions[j].tp++

            case classes[i] == j && prediction[i] != j:
                class_predictions[j].fn++
            }
        }

        class_predictions[j].precision = float64(class_predictions[j].tp) /
            (float64(class_predictions[j].tp) + float64(class_predictions[j].fp))

        class_predictions[j].recall = float64(class_predictions[j].tp) /
            (float64(class_predictions[j].tp) + float64(class_predictions[j].fn))

        class_predictions[j].f1 = 2.0 * (class_predictions[j].precision * class_predictions[j].recall) /
                                  (class_predictions[j].precision + class_predictions[j].recall)
    }

    var avg_precision, avg_recall, avg_f1 float64
    for j := 0; j < num_classes; j++ {
        avg_recall += class_predictions[j].recall * float64(class_predictions[j].support)
        avg_precision += class_predictions[j].precision * float64(class_predictions[j].support)
        avg_f1 += class_predictions[j].f1 * float64(class_predictions[j].support)
    }

    avg_f1 /= float64(len(classes))
    avg_precision /= float64(len(classes))
    avg_recall /= float64(len(classes))

    return avg_precision, avg_recall, avg_f1
}
