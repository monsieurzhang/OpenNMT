local model_utils = require 'lib.utils.model_utils'

local Memory = {}

local function _tensorIncluded(t, l)
  if torch.isTensor(l) then
    return torch.pointer(t:storage()) == torch.pointer(l:storage())
  elseif torch.type(l) == 'table' then
    for _, m in ipairs(l) do
      if _tensorIncluded(t, m) then return true end
    end
  end
  return false
end

-- we cannot share a tensor if it is exposed outside of the net - so cannot be part of output/gradInput
local function _canShare(t, net, netGradOutput)
  if torch.isTensor(t) and t:storage() then
    if not _tensorIncluded(t, net.gradInput) and not _tensorIncluded(t, netGradOutput) then
      return true
    end
  elseif torch.type(t) == 'table' then
    for _, m in ipairs(t) do
      if not _canShare(m, net, netGradOutput) then
        return false
      end
    end
    return true
  end
  return false
end

local function _size(t)
  local size=0
  if torch.isTensor(t) then
    if t:storage() then return t:storage():size()*t:elementSize() end
  elseif torch.type(t) == 'table' then
    for _, m in ipairs(t) do
      size = size + _size(m)
    end
  end
  return size
end

function Memory.optimize(model, criterion, batch)
  -- record actual size of the batch
  local actual_batchsize = { source_length = batch.source_length, target_length = batch.target_length }

  -- batch of one single word since we optimize the first clone
  batch.source_length = 1
  batch.target_length = 1

  local model_desc = {}

  for name, mod in pairs(model) do
    model_desc[name] = {}
    local net
    if mod.net then
      net = mod:net(1)
    else
      net = mod.network
    end
    model_desc[name]['net'] = net
  end

  -- initialize the network with a first batch
  local enc_states, context = model.encoder:forward(batch)
  local dec_outputs = model.decoder:forward(batch, enc_states, context)
  dec_outputs = model_utils.recursiveClone(dec_outputs)
  local enc_grad_states_out, grad_context, loss = model.decoder:backward(batch, dec_outputs, criterion)
  model.encoder:backward(batch, enc_grad_states_out, grad_context)

  local totSize = 0
  local sharedSize = 0
  local idx = 1
  for name, desc in pairs(model_desc) do
    local net = desc['net']
    net:apply(function(m)
      totSize = totSize + _size(m.gradInput)
      totSize = totSize + _size(m.output)
      if _canShare(m.gradInput,net) then
        sharedSize = sharedSize + _size(m.gradInput)
        m.gradInputSharedIdx = idx
        idx = idx + 1
      end
      if _canShare(m.output,net) then
        sharedSize = sharedSize + _size(m.output)
        m.outputSharedIdx = idx
        idx = idx + 1
      end
    end)
  end

  print("Memory Optimization - memory shared between clones: "..sharedSize.."/"..totSize.." bytes")

  -- restore batch
  batch.source_length = actual_batchsize.source_length
  batch.target_length = actual_batchsize.target_length
end

return Memory
