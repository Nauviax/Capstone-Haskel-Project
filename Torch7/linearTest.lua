---@diagnostic disable: undefined-global
-- Run with luajit "file"

require('torch')
require('nn')
require('optim')

-- Reads in a csv from given path, and returns a tensor object
function read_csv(csv_path)-- Reads in a csv from given path, and returns a tensor object
   data = {}
   first_line = true -- Skip header for now
   for line in io.lines(csv_path) do
      if first_line then
         first_line = false
         goto continue -- Skip first line, the header.
      end
      local val_numbs = {}
      for val in string.gmatch(line, "(%d+%.?%d*)") do -- Matches floats basically, as I couldn't find a split(',') in lua
         val_numbs[#val_numbs+1] = tonumber(val) -- Convert strings to numbers
      end
      data[#data+1] = val_numbs -- Save numbers to data
      ::continue::
   end
   -- Convert to tensor
   tensor = torch.Tensor(data)
   return tensor
 end

tensor = read_csv('diabetesCSV.csv')

-- Split the data into training and testing sets, 60% training, 40% testing
num_samples = tensor:size(1)
num_inputs = tensor:size(2) - 1
num_train = math.floor(num_samples * 0.6)
num_test = num_samples - num_train

train_set = tensor:narrow(1, 1, num_train)
test_set = tensor:narrow(1, num_train+1, num_test)

-- !!!
-- inputs = tensor:narrow(2, 1, num_inputs)
-- targets = tensor:narrow(2, num_inputs+1, 1)

-- Define the model (predictor)
model = nn.Sequential()
model:add(nn.Linear(num_inputs, 1))
--print(model:__tostring());

-- Define the loss function
criterion = nn.MSECriterion()

-- Train the model
-- function torch.DoubleTensor:size() return self:numel() end -- !!! Test to try and get rid of error

trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.
print(train_set[1])
print(train_set:size())
trainer:train(train_set)

