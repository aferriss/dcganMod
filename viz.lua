require 'image'
require 'nn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

net = util.load("fingerCheckpoints/experiment1_500_net_G.t7", 0)

print(net.modules)
noise = torch.Tensor(64,3,128,128)
noise:normal(0,1)
util.optimizeInferenceMemory(net)

-- local imgs = net:forward(noise)
-- local back = torch.rand(imgs:size()):fill(0)
-- print (back:size())
-- print(noise:size())
-- image.save("filtersForward.png", image.toDisplayTensor{input=imgs})
l = image.lena()

function vis( )
	filters = net:get(16).weight
	filters = filters:reshape(filters:size(1),filters:size(2), 4, 4)
	--filters = filters:narrow(2, 1, 3):resize(1,3,32,32)
	
	n = nn.SpatialConvolution(1, 3, 4, 4)
	-- print (n.weight)
	local total = torch.Tensor(3, 509, 509)
	for i = 1, 64 do
		filt = filters[i]
		n.weight = filt
		print(n.weight:size())
		res = n:forward((l))
		print(res:size())
		total = total:add(res)
		
	end
	image.save("lenaFilter.png", image.toDisplayTensor{input=total})

	
	--print(filters:size())
	-- for i=1, 10 do
	-- filterImg = image.toDisplayTensor{input=filters[i]}
	-- -- print(filterImg:size())
	-- --scaled = image.scale(filterImg, filterImg:size(3)*3,filterImg:size(2)*3, bicubic )
	-- local num = string.format("%05d", i)
	-- image.save("vizFilters/filters".. i ..".png",filterImg )
	-- end	
end

vis()