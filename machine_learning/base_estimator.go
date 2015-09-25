
package machine_learning


type BaseEstimator interface {
    PredictProbe(probe []float64) int
}
