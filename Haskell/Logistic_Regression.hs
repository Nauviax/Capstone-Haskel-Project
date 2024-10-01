module Logistic_Regression where
    import ML_Common (DataPoint, getXs, getYstr) -- Importing the DataPoint definition and helper functions.

    import System.Random (randomRs, newStdGen) -- "cabal install --lib random"
    import Data.List (transpose, elemIndex)
    import Data.Maybe (fromJust, fromMaybe)
    -- import Debug.Trace (trace) -- For debugging purposes

    -- Define the logistic regression model
    data LogisticRegression = LogisticRegression {
        weights :: [[Double]],
        bias :: [Double]
    }

    -- Initialize the model with random weights and bias
    initializeModel :: Int -> Int -> IO LogisticRegression
    initializeModel numFeatures numClasses = do
        gen <- newStdGen
        let ws = take (numFeatures * numClasses) $ randomRs (-0.01, 0.01) gen
        let bs = take numClasses $ randomRs (-0.01, 0.01) gen
        return $ LogisticRegression (chunksOf numFeatures ws) bs
        where -- chunksOf function to split random weights into lists for each class
            chunksOf _ [] = []
            chunksOf n xs = take n xs : chunksOf n (drop n xs)

    -- Softmax function
    softmax :: [Double] -> [Double]
    -- softmax xs = map (\x -> exp x / sumExpXs) xs -- Old code for reference
    --     where sumExpXs = sum (map exp xs)
    softmax xs = map (\x -> exp (x - maxX) / sumExpXs) xs
        where
            maxX = maximum xs -- Used to avoid randomly getting large and NaN-ing
            sumExpXs = sum (map (exp . subtract maxX) xs)

    -- Training function using gradient descent
    train :: [DataPoint] -> Int -> Double -> Int -> IO LogisticRegression
    train dataPoints numClasses learningRate epochs = do -- Takes data points, number of classes, learning rate, and epochs as input
        let numFeatures = length (getXs (head dataPoints))
        model <- initializeModel numFeatures numClasses
        let labels = map getYstr dataPoints
        let uniqueLabels = unique labels
        let yIndices = map (\y -> fromJust $ elemIndex y uniqueLabels) labels
        return $ gradientDescent model dataPoints yIndices learningRate epochs -- Return the trained model
        where
            unique = foldl (\seen x -> if x `elem` seen then seen else seen ++ [x]) [] -- Get unique elements

            gradientDescent model dataPoints yIndices lr 0 = model -- Stop when epoch is 0
            gradientDescent model dataPoints yIndices lr epoch = -- Update weights and bias while epoch is not 0
                let (newWeights, newBias) = updateWeightsAndBias model dataPoints yIndices lr
                in gradientDescent (LogisticRegression newWeights newBias) dataPoints yIndices lr (epoch - 1) -- Recurse with updated model

            -- From what I can tell, below error calculation is 100% correct, but the weights and bias are somehow updating in a way that makes the error LARGER next epoch.

            updateWeightsAndBias (LogisticRegression ws bs) dataPoints yIndices lr = -- Update weights and bias (Takes the model, datapoints, y indices, and learning rate as input)
                let predictions = map (predictProbs (LogisticRegression ws bs) . getXs) dataPoints -- Predict probabilities for each data point
                    errors = zipWith (\pred yIdx -> zipWith (-) pred (oneHot yIdx numClasses)) predictions yIndices -- Calculate errors by subtracting one-hot vectors from predictions
                    weightGradients = transpose $ zipWith (zipWith (*)) (map getXs dataPoints) errors -- Calculate weight gradients by multiplying Xs with errors
                    biasGradients = map sum (transpose errors) -- Calculate bias gradients by summing errors
                    newWeights = zipWith (zipWith (\ww grad -> ww - lr * grad)) ws weightGradients -- Update weights (ws) using the gradients and learning rate (lr)
                    newBias = zipWith (\bb grad -> bb - lr * grad) bs biasGradients -- Update bias (bs) using the gradients and learning rate
                in (newWeights, newBias) -- Return updated weights and bias

            oneHot idx len = replicate idx 0 ++ [1] ++ replicate (len - idx - 1) 0 -- Create one-hot vector, e.g. oneHot 2 5 = [0, 0, 1, 0, 0]

    -- Predict probabilities for a new set of Xs
    predictProbs :: LogisticRegression -> [Double] -> [Double]
    predictProbs (LogisticRegression ws bs) xs = softmax $ zipWith (+) (map (sum . zipWith (*) xs) ws) bs

    -- Predict the class for a new set of Xs
    predict :: LogisticRegression -> [String] -> [Double]  -> String
    predict model uniqueLabels xs = uniqueLabels !! maxIndex
        where
            probs = predictProbs model xs
            maxIndex = fromJust $ elemIndex (maximum probs) probs
            -- maxIndex = fromMaybe (error (printf "Maximum probability not found. Probs: %s, Maximum: %f" (show probs) (maximum probs))) $ elemIndex (maximum probs) probs

    -- Calculate accuracy of the model
    calculateAccuracy :: [String] -> [String] -> Double -- Takes actual then predicted labels as input
    calculateAccuracy actuals predictions = correct / total
        where
            correct = fromIntegral $ length $ filter (uncurry (==)) (zip actuals predictions) -- Calculate number of actuals == predictions
            total = fromIntegral $ length actuals