---@diagnostic disable: undefined-global
-- Copy to wherever neeeded. Look into some way to require this file maybe?
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

  -- -- Print table as test
  -- for index, row in pairs(data) do
  --    print(table.concat(row, ", "))
  -- end

  -- Convert to tensor
  tensor = torch.Tensor(data)
  return tensor
end