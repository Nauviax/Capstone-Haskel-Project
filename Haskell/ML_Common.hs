module ML_Common where -- Items used in all other files, like DataPoint definitions.
    import System.Random.Shuffle (shuffleM) -- "cabal install --lib random-shuffle"
    import Data.Either (lefts) -- Because I decided to support strings as y values.

    -- Holds a list of x features and a single y value.
    type YValue = Either Double String
    data DataPoint = DataPoint { xs :: [Double], y :: YValue } deriving (Show)

    -- Helper functions to get the x features and y value of a DataPoint.
    getXs :: DataPoint -> [Double]
    getXs (DataPoint xs _) = xs
    getXat :: DataPoint -> Int -> Double
    getXat (DataPoint xs _) attributeIndex = xs !! attributeIndex
    getYnum :: DataPoint -> Double
    getYnum (DataPoint _ (Left y)) = y
    getYstr :: DataPoint -> String
    getYstr (DataPoint _ (Right y)) = y
    getYstr (DataPoint _ (Left y)) = show y -- !!! Hack fix for integer class values in some datasets
    

    -- Reads a CSV file and converts each line into a dataPoint.
    -- Each line is split by commas, converted to a list of Doubles, and then to a dataPoint.
    readCSV :: FilePath -> IO [DataPoint]
    readCSV filePath = do
        content <- readFile filePath
        let linesOfFiles = tail $ lines content -- Drop first line, which is the header
        return $ map (toPoint . map safeRead . splitString (==',')) linesOfFiles
        where
            toPoint :: [Either Double String] -> DataPoint
            toPoint xs = DataPoint (lefts (init xs)) (last xs) -- Splits the list into features (xs) and label (y).

            splitString :: (Char -> Bool) -> String -> [String]
            splitString p s =  case dropWhile p s of
                "" -> []
                s' -> w : splitString p s'' -- Splits a string into a list of strings based on a predicate.
                    where (w, s'') = break p s'

            safeRead :: String -> Either Double String -- Prevent "Prelude.read: no parse" error.
            safeRead s = case reads s of
                [(x, "")] -> Left x -- Successfully parsed the entire string as Double.
                _ -> Right (filter (`notElem` "\\\"") s) -- If not a Double, remove \ and " from the string.

    -- Splits a list into training and testing sets based on a given ratio.
    -- The data is shuffled before splitting to ensure randomness.
    splitData :: [DataPoint] -> Double -> IO ([DataPoint], [DataPoint])
    splitData dataList trainSize = do
        let splitIndex = floor $ trainSize * fromIntegral (length dataList) -- Calculates the split index.
        shuffledData <- shuffleM dataList -- Shuffles the data to randomize it.
        return (splitAt splitIndex shuffledData) -- Splits the data.