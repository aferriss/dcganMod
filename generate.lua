require 'image'
require 'nn'
local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 32,        -- number of samples to produce
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'random',  -- random / line / linefull1d / linefull
    name = 'generation1',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 1,           -- Display image: 0 = false, 1 = true
    nz = 100,              
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(net ~= '', 'provide a generator model')

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
net = torch.load(opt.net)

-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(net:get(1)) == 'nn.View' then
    net:remove(1)
end

print(net)

if opt.noisetype == 'uniform' then
    noise:uniform(-1, 1)
elseif opt.noisetype == 'normal' then
    noise:normal(0, 1)
end

noiseL = torch.FloatTensor(opt.nz):uniform(-1, 1)
noiseR = torch.FloatTensor(opt.nz):uniform(-1, 1)

-- ADAM CODE 

function round(num, idp)
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end
print "bouta get uff"

function OneHot(X, n, negative_class)
    xSize = X:size(1)
    Xoh = torch.Tensor(xSize, n):fill(0)
    lin = torch.linspace(1, xSize, xSize)
    for j = 1, X:size(1) do
        Xoh[j][X[j]+1] = 1 
    end

    return Xoh
end

ny = 100
function get_buffer_y(steps, num_buffer_samples, num_buffer_steps)
    num_buffer_rows = math.floor( math.ceil( num_buffer_samples / steps ))
    targets = torch.Tensor(num_buffer_rows, steps):zero()
    for _ = 0, num_buffer_rows-1 do
        for i = 0, steps-1 do
            targets[_+1][i+1] = math.floor(round(_*steps+i/num_buffer_steps, 1))
        end
    end

    targets = targets:view(targets:nElement())

    
    ymb = OneHot(targets, ny, 0)
    return ymb
end

--

if opt.noisemode == 'line' then
   -- do a linear interpolation in Z space between point A and point B
   -- each sample in the mini-batch is a point on the line
    line  = torch.linspace(0, 1, opt.batchSize)
    for i = 1, opt.batchSize do
        noise:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull1d' then
   -- do a linear interpolation in Z space between point A and point B
   -- however, generate the samples convolutionally, so a giant image is produced
    assert(opt.batchSize == 1, 'for linefull1d mode, give batchSize(1) and imsize > 1')
    noise = noise:narrow(3, 1, 1):clone()
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull' then
   -- just like linefull1d above, but try to do it in 2D
    assert(opt.batchSize == 1, 'for linefull mode, give batchSize(1) and imsize > 1')
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(3, i, 1):narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end


--[[

local sample_input = torch.randn(2,100,1,1)
if opt.gpu > 0 then
    net:cuda()
    cudnn.convert(net, cudnn)
    noise = noise:cuda()
    sample_input = sample_input:cuda()
else
   sample_input = sample_input:float()
   net:float()
end

-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
optnet.optimizeMemory(net, sample_input)

local images = net:forward(noise)
print('Images size: ', images:size(1)..' x '..images:size(2) ..' x '..images:size(3)..' x '..images:size(4))
images:add(1):mul(0.5)
print('Min, Max, Mean, Stdv', images:min(), images:max(), images:mean(), images:std())
image.save(opt.name .. '.png', image.toDisplayTensor(images))
print('Saved image to: ', opt.name .. '.png')

if opt.display then
    disp = require 'display'
    disp.image(images)
    print('Displayed image')
end

--]]



    print(noise:size())
elseif opt.noisemode == 'adam' then

    numRand = 80
    classes = {}
    for i = 1, numRand do
        table.insert(classes,i,i)
    end

    table.insert(classes, table.getn(classes), 1)
    -- table.insert(classes, table.getn(classes)-1, 1)

    --classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1}
    
    --classes = {1, 2, 3, 4, 5, 1, 1}
    steps = 55 
    --num_buffer_classes = 1
    --bymb = get_buffer_y(steps, num_buffer_classes, 1)
    offset = 0--bymb:size(1)


    numTargets = table.getn(classes)
    targets = torch.Tensor(numTargets, steps):zero()
    for _ = 0, numTargets-1 do
        for i = 0, steps-1 do
            targets[_+1][i+1] = classes[i+1]
        end
    end 

    ymb = OneHot(targets:view(targets:nElement()), ny)

    -- ymb = torch.cat(bymb, ymb, 1)


    for i = 1, numTargets do
        y1 = classes[i]

        y2 = classes[math.fmod(((i)+1), numTargets)]
        if y2 == nil then
             y2 = classes[1]
        end
        -- print (y2)
        for j =1, steps do
            
            y = 1 + steps * (i-1) + (j-1)
             if y < ymb:size(1) then
                ymb[y] = torch.Tensor(ny):fill(0)
                if y1 == y2 then
                    ymb[y] = torch.Tensor(ny):zero()
                else
                    -- print(y)
                    ymb[y][y1] = 1.0 - (j-1) / (steps-1.0)
                    ymb[y][y2] = (j-1) / (steps - 1.0)
                end
             end
        end
        -- print(y2)
    end

     
    ymb = ymb:resize(ymb:size(1)-steps*2, ymb:size(2), 1, 1)
    -- print (offset)
    print(ymb:size())
    noise = ymb
end
local inc = 0


function feedAndSave()
    if opt.gpu > 0 then
    net:cuda()
    cudnn.convert(net, cudnn)
    noise = noise:cuda()
    --sample_input = sample_input:cuda()
        -- ymb = ymb:cuda()
    else
       net:float()
    end

    -- a function to setup double-buffering across the network.
    -- this drastically reduces the memory needed to generate samples
    --util.optimizeInferenceMemory(net)

    local images = net:forward(noise)
    --images = images:view(images:size(1), 1, images:size(2), images:size(3))
    print(images:size())
    print('Images size: ', images:size(1)..' x '..images:size(2) ..' x '..images:size(3)..' x '..images:size(4))
    --images:add(1):mul(0.5)
    --print('Min, Max, Mean, Stdv', images:min(), images:max(), images:mean(), images:std())

    local ts = os.time()
    local tss = string.format(os.date('%Y-%m-%d-%H-%M-%S', ts))
    local fileCount = string.format("%05d", inc)

--    image.save("savedImages/" .. tss .. '.png', image.toDisplayTensor({input=images}))
    image.save( tss .. '.png', image.toDisplayTensor({input=images}))
    print('Saved image to: ', opt.name .. '.png')
--    os.execute("mkdir " .. "savedImages/frames/" .. tss)
    if opt.noisemode == 'adam' then
        for i = 1, images:size(1) do
            num = string.format("%05d", i)
            image.save(num .. '.png', image.toDisplayTensor(images[i]))
        end
    end 

    if opt.display then
        disp = require 'display'
        disp.image(images,{win=1000, title="blah"})
        print('Displayed image')
    end

    --addNoise()
end





function addNoise()
    -- print(noise:size())
    --print(noise[1][1][1]:size())
    local valToAdd = torch.Tensor(noise:size(3) , noise:size(4)):fill(0.005)
    local valToSub = torch.Tensor(noise:size(3) , noise:size(4)):fill(-1.0)
    valToAdd = valToAdd:cuda()
    --print(noise[1][1]:size())
    for j =1, noise:size(1) do
        for i=1, noise:size(2) do
            noise[j][i]:add(valToAdd)
            -- local noiseMax = torch.max(noise[1][i])
            -- if noiseMax > 10 then
            --  noise[1][i]:add(valToSub)
            -- end
            --noise[1][i]:mod(10)

            for y=1, noise:size(3) do
                for x=1, noise:size(4) do
                    --noise[1][i][y][x] = noise[1][i][y][x] + 0.01

                    if noise[j][i][y][x] > 3 then 
                        noise[j][i][y][x] = -2-- torch.normal()
                    end
                end
            end
        end
    end


    -- for i=1, opt.imsize do
    --  noise:narrow(3, i, 1):narrow(4, i, 1)[1][i][1][1] = noise:narrow(3, i, 1):narrow(4, i, 1)[1][i][1][1] + 0.1
    --  --local pos = noise:narrow(3, i, 1):narrow(4, i, 1)
    --  --print(pos)
    --  if  noise:narrow(3, i, 1):narrow(4, i, 1)[1][i][1][1] >= 2 then
    --      noise:narrow(3, i, 1):narrow(4, i, 1)[1][i][1][1] = -1
    --  end
    -- end
    inc = inc+1
    feedAndSave()
end

feedAndSave()
function vis()
    print(noise:size())
    -- noise = noise:view(noise:size(1))
end
