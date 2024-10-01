import ML_Common (readCSV, splitData, getXs, getYnum, getYstr) -- Importing the DataPoint definition and helper functions.

import Linear_Regression (train, predict, calculateRSquare) 
import Logistic_Regression (train, predict, calculateAccuracy)
import KNN (train, predict, calculateAccuracy)
import Regression_Tree (train, predict, calculateRSquare, treeToString)

import Data.Time (diffUTCTime, getCurrentTime)
import GHC.Stats (getRTSStats, GCDetails(..), RTSStats(..))
import Data.List (nub) -- Unique elements in a list
import Control.Monad (when) -- For conditional execution (Seemed cooler than if-else at the time)

-- Define boolean flags for each model (Edit these to run different models)
runLinearRegression :: Bool
runLogisticRegression :: Bool
runKNN :: Bool
runRegressionTree :: Bool
runLinearRegression = False
runLogisticRegression = True
runKNN = False
runRegressionTree = False

main :: IO ()
main = do
    startTime <- getCurrentTime
    startStats <- getRTSStats
    putStrLn "Starting model execution..." -- After testing with traces, this actually seems to print AFTER model training etc. (So it's technically lying)

    -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% --
    when runLinearRegression $ do
        putStrLn "Linear Regression:"

        allPoints <- readCSV "D:/_HaskelProject/CSV/house8L_CSV.csv" -- Reads points from a CSV file.
        let trainSize = 0.6 -- Sets the training data size to 60%.
        (trainPoints, testPoints) <- splitData allPoints trainSize -- Splits the data into training and testing sets.

        let coefficients = Linear_Regression.train trainPoints -- Calculates the coefficients using linear regression.
        putStrLn "Coefficients:"
        print coefficients

        let predicted = map (Linear_Regression.predict coefficients . getXs) testPoints -- Predicts the y values for the test set.
        let actual = map getYnum testPoints -- Extracts the actual y values of the test set.
        -- putStrLn "Predictions (X val, Y pred, Y actual):" -- Will only show first x value for brevity.
        -- mapM_ print $ zipWith3 (\p yPred yActual -> (head (getXs p), yPred, yActual)) testPoints predicted actual
        
        let rSquare = Linear_Regression.calculateRSquare actual predicted -- Calculates the R-Squared value.
        putStrLn "R-Square:"
        print rSquare

    -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% --
    when runLogisticRegression $ do
        putStrLn "Logistic Regression:"

        allPoints <- readCSV "D:/_HaskelProject/CSV/spambaseCSV.csv" -- Reads points from a CSV file.
        let uniqueLabels = nub $ map getYstr allPoints -- Get unique Y labels from the dataset.
        let trainSize = 0.6 -- Sets the training data size to 60%.
        (trainPoints, testPoints) <- splitData allPoints trainSize -- Splits the data into training and testing sets.

        model <- Logistic_Regression.train trainPoints (length uniqueLabels) 0.02 400 -- Trains the model using logistic regression.

        let predicted = map (Logistic_Regression.predict model uniqueLabels . getXs) testPoints -- Predicts the y values for the test set.
        let actual = map getYstr testPoints -- Extracts the actual y values of the test set.
        -- putStrLn "Predictions (X val, Y pred, Y actual):" -- Will only show first x value for brevity.
        -- mapM_ print $ zipWith3 (\p yPred yActual -> (head (getXs p), yPred, yActual)) testPoints predicted actual

        let accuracy = Logistic_Regression.calculateAccuracy actual predicted -- Calculates the accuracy of the model.
        putStrLn "Accuracy:"
        print accuracy

    -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% --
    when runKNN $ do
        putStrLn "K-Nearest Neighbors (KNN):"

        allPoints <- readCSV "D:/_HaskelProject/CSV/spambaseCSV.csv" -- Reads points from a CSV file.
        let uniqueLabels = nub $ map getYstr allPoints -- Get unique Y labels from the dataset.
        let trainSize = 0.6 -- Sets the training data size to 60%.
        (trainPoints, testPoints) <- splitData allPoints trainSize -- Splits the data into training and testing sets.

        let knnModel = KNN.train trainPoints 6 -- Trains the KNN model with k = 6.

        -- let predicted = map (KNN.predict knnModel) testPoints -- Predicts the y values for the test set.
        -- let actual = map getYstr testPoints -- Extracts the actual y values of the test set.
        -- putStrLn "Predictions (X val, Y pred, Y actual):" -- Will only show first x value for brevity.
        -- mapM_ print $ zipWith3 (\p yPred yActual -> (head (getXs p), yPred, yActual)) testPoints predicted actual

        let accuracy = KNN.calculateAccuracy knnModel testPoints -- Calculates the accuracy of the model.
        putStrLn "Accuracy:"
        print accuracy

    -- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% --
    when runRegressionTree $ do
        putStrLn "Regression Tree:"

        allPoints <- readCSV "D:/_HaskelProject/CSV/house8L_CSV.csv" -- Reads points from a CSV file.
        let trainSize = 0.6 -- Sets the training data size to 60%.
        (trainPoints, testPoints) <- splitData allPoints trainSize -- Splits the data into training and testing sets.

        let treeModel = Regression_Tree.train 8 trainPoints -- Trains the regression tree model.

        let predicted = map (Regression_Tree.predict treeModel . getXs) testPoints -- Predicts the y values for the test set.
        let actual = map getYnum testPoints -- Extracts the actual y values of the test set.
        -- putStrLn "Predictions (X val, Y pred, Y actual):" -- Will only show first x value for brevity.
        -- mapM_ print $ zipWith3 (\p yPred yActual -> (head (getXs p), yPred, yActual)) testPoints predicted actual

        let rSquare = Regression_Tree.calculateRSquare actual predicted -- Calculates the R-Squared value.
        putStrLn "R-Square:"
        print rSquare

        -- let treeString = Regression_Tree.treeToString treeModel -- Converts the tree model to a string.
        -- putStrLn "Tree Structure:"
        -- putStrLn treeString

    endTime <- getCurrentTime
    endStats <- getRTSStats
    let elapsedTime = diffUTCTime endTime startTime
    let memoryUsed = gcdetails_live_bytes (gc endStats) - gcdetails_live_bytes (gc startStats)
    let memoryUsedMB = fromIntegral memoryUsed / (1024 * 1024)
    putStrLn $ "Execution time: " ++ show elapsedTime
    putStrLn $ "Memory used: " ++ show memoryUsedMB ++ " MB"