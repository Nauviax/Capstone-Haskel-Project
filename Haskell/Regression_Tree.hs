module Regression_Tree where
    -- After COMPX521 made me do so much java stuff with regression trees and XGBoost, might as well do it in Haskell too.
    import ML_Common (DataPoint, getXs, getXat, getYnum) -- Importing the DataPoint definition and helper functions.

    import Data.List (sortOn)
    -- import Debug.Trace (trace) -- For debugging purposes

    -- | Constant value for negative infinity.
    negInf :: Double
    negInf = negate 1/0

    -- | Represents a node in the decision tree. Attribute, split point, then left and right children for internal node.
    data Node = InternalNode Int Double Node Node
              | LeafNode Double

    -- | A class for objects that hold a split specification, including the quality of the split.
    data SplitSpecification = SplitSpecification {
        attributeIndex :: Int, -- Index for getXs
        splitPoint :: Double,
        splitQuality :: Double
    } deriving (Show)

    -- Contains the stats required to measure split quality.
    data SufficientStatistics = SufficientStatistics {
        num :: Int,
        sumOfVals :: Double,
        sumOfSquares :: Double
    } deriving (Show)

    -- | Updates sufficient statistics based on a class value and whether it is being added or removed.
    updateStats :: SufficientStatistics -> Double -> Bool -> SufficientStatistics
    updateStats stats value isRight =
        if isRight
        then stats { num = num stats + 1, sumOfVals = sumOfVals stats + value, sumOfSquares = sumOfSquares stats + value * value }
        else stats { num = num stats - 1, sumOfVals = sumOfVals stats - value, sumOfSquares = sumOfSquares stats - value * value }

    -- | Computes the sum of squared deviations from the mean based on the sufficient statistics provided.
    -- | sosdftm = Sum of Squares Deviations From The Mean
    sosdftm :: SufficientStatistics -> Double
    sosdftm stats = sumOfSquares stats - (sumOfVals stats * sumOfVals stats / fromIntegral (num stats))

    -- | Computes the reduction in the sum of squared errors based on the sufficient statistics provided.
    -- | The initialSufficientStatistics are the sufficient statistics based on the data before it is split,
    -- | statsLeft are the sufficient statistics for the left branch, similar for right.
    calcSplitQuality :: SufficientStatistics -> SufficientStatistics -> SufficientStatistics -> Double
    calcSplitQuality initialSufficientStatistics statsLeft statsRight =
        sosdftm initialSufficientStatistics - (sosdftm statsLeft + sosdftm statsRight)

    -- | Finds the best split point and returns the corresponding split specification object.
    findBestSplitPoint :: [DataPoint] -> Int -> SufficientStatistics -> SplitSpecification
    findBestSplitPoint dataPoints attributeIndex initialStats =
        let statsLeft = SufficientStatistics (num initialStats) (sumOfVals initialStats) (sumOfSquares initialStats)
            statsRight = SufficientStatistics 0 0.0 0.0
            splitSpecification = SplitSpecification attributeIndex 0.0 negInf
            previousValue = negInf
            -- sortedIndices = map fst . sortOn (getXs . snd) $ zip [0..] dataPoints -- Old version
            sortedIndices = map fst . sortOn (\dp -> getXat (snd dp) attributeIndex) $ zip [0..] dataPoints -- MIGHT work better
            updateStatsLeftRight dp = (updateStats statsLeft (getYnum dp) False, updateStats statsRight (getYnum dp) True)
            processNextSplit (prevValue, bestSplitSpec, statsL, statsR) ii =
                let dataPoint = dataPoints !! ii
                    currentValue = getXat dataPoint attributeIndex
                    (newStatsL, newStatsR) = updateStatsLeftRight dataPoint
                    newSplitQuality = calcSplitQuality initialStats newStatsL newStatsR
                    newSplitSpec = if currentValue > prevValue && newSplitQuality > splitQuality bestSplitSpec && ((currentValue + prevValue) / 2.0) > negInf -- Unsure why this is required !!!
                                then bestSplitSpec { splitQuality = newSplitQuality, splitPoint = (currentValue + prevValue) / 2.0 }
                                else bestSplitSpec
                in (currentValue, newSplitSpec, newStatsL, newStatsR)
            (_, finalSplitSpec, _, _) = foldl processNextSplit (previousValue, splitSpecification, statsLeft, statsRight) sortedIndices
        in finalSplitSpec

    -- | Creates and returns a leaf node, given stats.
    createLeafNode :: SufficientStatistics -> Node
    createLeafNode stats = LeafNode (sumOfVals stats / fromIntegral (num stats))

    -- | Recursively grows a tree for a given set of data.
    makeTree :: [DataPoint] -> Int -> Node
    makeTree dataPoints maxDepth =
        let stats = foldl (\acc dp -> updateStats acc (getYnum dp) True) (SufficientStatistics 0 0.0 0.0) dataPoints
        in if num stats <= 1 || maxDepth == 0
        then createLeafNode stats -- Return leaf node if just one datapoint, or max depth reached
        else let bestSplitSpecification = foldl (\bestSpec attrIndex ->
                        let splitSpec = findBestSplitPoint dataPoints attrIndex stats
                        in if splitQuality splitSpec > splitQuality bestSpec then splitSpec else bestSpec
                    ) (SplitSpecification (-1) negInf negInf) [0..(length (getXs (head dataPoints)) - 1)]
                in if splitQuality bestSplitSpecification < 1E-6
                then createLeafNode stats -- Return leaf node if split quality is too low
                else let (leftSubset, rightSubset) = foldl (\(left, right) dp ->
                                if getXat dp (attributeIndex bestSplitSpecification) < splitPoint bestSplitSpecification
                                then (dp:left, right)
                                else (left, dp:right)
                            ) ([], []) dataPoints
                        in InternalNode (attributeIndex bestSplitSpecification) (splitPoint bestSplitSpecification)
                            (makeTree leftSubset (maxDepth - 1)) (makeTree rightSubset (maxDepth - 1))

    -- | Builds the tree classifier, and returns the root node. Takes a max depth as additional input.
    train ::  Int -> [DataPoint] -> Node
    train maxDepth trainingData  = makeTree trainingData maxDepth

    -- Recursive function to make a prediction, given a tree and xVals.
    predict :: Node -> [Double] -> Double
    predict (LeafNode prediction) _ = prediction
    predict (InternalNode attributeIndex splitPoint left right) xVals =
        if xVals !! attributeIndex < splitPoint
        then predict left xVals
        else predict right xVals

    -- Calculates the R-Squared value to evaluate the model performance. (Copied from Linear Regression)
    calculateRSquare :: [Double] -> [Double] -> Double
    calculateRSquare actual predicted =
        let meanActual = sum actual / fromIntegral (length actual) -- Calculates the mean of actual values.
            totalSumSquares = sum $ map (\x -> (x - meanActual) ** 2) actual -- Total sum of squares.
            residualSumSquares = sum $ zipWith (\a p -> (a - p) ** 2) actual predicted -- Residual sum of squares.
        in 1 - residualSumSquares / totalSumSquares -- Calculates R-Squared.

    -- Helper function to append indentation
    appendIndentation :: Int -> String
    appendIndentation level = concat (replicate level "|   ")

    -- Function to convert a branch to string
    branchToString :: String -> Bool -> Int -> Node -> String
    branchToString sb left level (InternalNode attributeIndex splitPoint leftNode rightNode) =
        let comparison = if left then " < " else " >= "
            newSb = sb ++ "\n" ++ appendIndentation level ++ "Attribute " ++ show attributeIndex ++ comparison ++ show splitPoint
        in toString newSb (level + 1) (if left then leftNode else rightNode)
    branchToString sb _ _ _ = sb -- This case should not happen

    -- Recursive function to convert a subtree to string
    toString :: String -> Int -> Node -> String
    toString sb level (LeafNode prediction) = sb ++ ": " ++ show prediction
    toString sb level node@(InternalNode {}) =
        let sbLeftSide = branchToString sb True level node
            sbBothSides = branchToString sbLeftSide False level node
        in sbBothSides

    -- Function to convert the entire tree to string
    treeToString :: Node -> String
    treeToString = toString "" 0