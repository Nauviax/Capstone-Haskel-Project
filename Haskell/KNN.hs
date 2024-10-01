module KNN where
    import ML_Common (DataPoint, getXs, getYstr) -- Importing the DataPoint definition and helper functions.

    import Data.List (sortBy, group, sort, maximumBy)
    import Data.Ord (comparing)

    -- Define the KNN model
    data KNN = KNN {
        trainingData :: [DataPoint],
        k :: Int
    }

    -- "Train" the KNN model (Should really be called "fit" or something, but this matches the other models)
    train :: [DataPoint] -> Int -> KNN
    train dataPoints kValue = KNN { trainingData = dataPoints, k = kValue }

    -- Predict the label for a new data point
    predict :: KNN -> DataPoint -> String
    predict knn newPoint = majorityVote $ take (k knn) nearestNeighbors
        where
            nearestNeighbors = sortBy (comparing (distance newPoint)) (trainingData knn)

    -- Calculate the accuracy of the KNN model
    calculateAccuracy :: KNN -> [DataPoint] -> Double
    calculateAccuracy knn testData = correct / total
        where
            total = fromIntegral $ length testData
            correct = fromIntegral $ length $ filter (uncurry (==)) predictions
            predictions = [(getYstr dp, predict knn dp) | dp <- testData]

    -- Calculate the distance between two data points (!!! Should I be normalizing the data or something? Or is that done in pre-processing?)
    distance :: DataPoint -> DataPoint -> Double
    distance dp1 dp2 = sqrt . sum $ zipWith (\x y -> (x - y) ^ 2) (getXs dp1) (getXs dp2)

    -- Determine the majority vote from the nearest neighbors. (Group by label and take the label with the most occurrences)
    majorityVote :: [DataPoint] -> String
    majorityVote neighbors = fst . maximumBy (comparing snd) . map (\l -> (head l, length l)) . group . sort $ map getYstr neighbors
