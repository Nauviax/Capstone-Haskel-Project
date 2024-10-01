module Linear_Regression where

    import ML_Common (DataPoint, getXs, getYnum) -- Importing the DataPoint definition and helper functions.
    import Data.Matrix (fromList, fromLists, inverse, toList, transpose) -- "cabal install --lib matrix"

    -- Performs linear regression on a list of Points to find the coefficients.
    train :: [DataPoint] -> [Double]
    train points =
        let
            xs = map (\p -> 1 : getXs p) points -- Prepends 1 to each feature vector for the intercept term.
            xMatrix = fromLists xs -- Constructs the design matrix from the feature vectors.
            y = fromList (length points) 1 (map getYnum points) -- Constructs a column matrix of y values.
            theta = case inverse (Data.Matrix.transpose xMatrix * xMatrix) of
                Right inv -> (inv * Data.Matrix.transpose xMatrix) * y -- The actual rest of getting theta
                Left errorMsg -> error $ "Matrix inversion failed: " ++ errorMsg -- AAAAA why are you like this inverse
        in toList theta -- Converts the matrix of coefficients to a list.

    -- Predicts a value given a list of coefficients and a list of features.
    -- The prediction is the dot product of the coefficients and the features, including the intercept term.
    predict :: [Double] -> [Double] -> Double
    predict coefficients x = sum $ zipWith (*) coefficients (1 : x) -- Prepend 1 to the features for the intercept term.

    -- Calculates the R-Squared value to evaluate the model performance.
    calculateRSquare :: [Double] -> [Double] -> Double
    calculateRSquare actual predicted =
        let meanActual = sum actual / fromIntegral (length actual) -- Calculates the mean of actual values.
            totalSumSquares = sum $ map (\x -> (x - meanActual) ** 2) actual -- Total sum of squares.
            residualSumSquares = sum $ zipWith (\a p -> (a - p) ** 2) actual predicted -- Residual sum of squares.
        in 1 - residualSumSquares / totalSumSquares -- Calculates R-Squared.