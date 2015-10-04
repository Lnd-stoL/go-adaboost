
package machine_learning


type BaseClassifier interface {
    PredictProbe(probe []float64) int
}
