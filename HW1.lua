-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'SST1.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-dev', 'false', 'narrow training data for development')

-- Hyperparameters
cmd:option('-lambda', 2, 'Regularization term parameter')
cmd:option('-alpha', 1, 'extra count for words')
cmd:option('-batchsize', 50, 'Size of the Minibatch for SGD')
cmd:option('-learningrate', .02, 'Learning Rate')
cmd:option('-epochs', 10, 'Number of Epochs')

-- ...

function linearPredictor(W, b, x, y, inputindices, accuracy)
  -- inputs are tensor W and tensor b to compute the prediction f(Wx+b)
  local yhat = torch.DoubleTensor(x:size()[1]):fill(0)
  for c = 1, x:size()[1], 1 do
    local row = x[c]

    --access all the indices of the sparse matrix and compute Wx+ b
    --vindices is the tensor with indices of last non 1 element
    local Wxb = b:clone()
    for n=1, inputindices[c] do
      Wxb = Wxb + W[row[n]]
    end

    --make a list of all the highest and pick a random one from them
    local hindex = {}
    local hscore = -math.huge
    for n=1, Wxb:size()[1] do
      if Wxb[n] > hscore then
        hindex = {}
        hindex[1] = n
        hscore = Wxb[n]
      elseif Wxb[n] == hscore then
        table.insert(hindex, n)
      end
    end
    local pick = hindex[math.random(#hindex)]
    --to have the difference in prediction right away
    yhat[c] = pick -- outputs[c]

  end
  if accuracy == true then
    comp_accuracy(yhat, y)
  end
  return yhat
end

function naiveBayes(W, b)
  --step one, create correct W and b
  b:fill(0)
  W:fill(0)

  --count all the words
  local wordcount = torch.Tensor(nclasses)
  local alpha = opt.alpha
  W:fill(alpha)
  b:fill(alpha)
  for row=1, tin:size()[1] do
    if row % 10000 == 0 then
      print(row, "rows counted")
    end
    b[tout[row]] = b[tout[row]] + 1
    for char=1, tindices[row] do
      local class = tout[row]
      wordcount[class] = wordcount[class] + 1
      --W [feature] [class]
      --current feature is: tin[row][char]
      W[tin[row][char]][class] = W[tin[row][char]][class] + 1
    end
  end
  --get correct fraction
  for row=1, W:size()[1] do
    for c=1, nclasses do

      W[row][c] = torch.log(W[row][c] / wordcount[c])
    end
  end

  for row=1, nclasses do
    b[row] = torch.log(b[row] / tout:size()[1])
  end
  -- predict y
  local yhat = linearPredictor(W, b, tin, tout, tindices, true)

  return W, b
end

function logisticReg(W, b, x, y, inputindices)
  --step 1: compute scores
  local z = torch.DoubleTensor(x:size()[1], nclasses)
  for row=1, 1 do--x:size()[1] do
    local scores = b:clone()
    for n=1, inputindices[row] do
      -- Scores of current row of current word index
      scores = scores + W[x[row][n]]
    end
    z[row] = scores
  end

  --step 2: compute softmax (efficient)
  local yhat = logsoftmax(z)


  --step 3: compute Loss
  local l = loss(yhat, y) + l2reg(W)
  print(l, "Loss")

  --step 4: compute Gradients
  W,b =sgd(x, y, W, b, loss, gradients_logr)
  return W,b
end

function hinge(W, b, x, y, inputindices)
  --step 1: compute scores
  local yhat = torch.DoubleTensor(x:size()[1], nclasses)
  for row=1, 1 do--x:size()[1] do
    local scores = b:clone()
    for n=1, inputindices[row] do
      -- Scores of current row of current word index
      scores = scores + W[x[row][n]]
    end
    yhat[row] = scores
  end


  --step 3: compute Loss
  local l = hloss(yhat, y) + l2reg(W)
  print(l, "Loss")

  --step 4: compute Gradients
  W,b = sgd(x, y, W, b, hloss, gradients_hinge)
  return W,b
end

-- Helper Functions -------------------------------------------------
function loss(yhat, y)
  local l = 0
  for row=1, yhat:size()[1] do 
    l = l - yhat[row][y[row]]
  end
  return l
end

function hloss(yhat, y)
  local l = 0
  for row=1, yhat:size()[1] do
    --get highest non true score 
    local highest_nontrue = 0 
    local counter = 1
    for pred=1,yhat[row]:size()[1] do
      if counter ~= y[row] and yhat[row][pred] > highest_nontrue then
        highest_nontrue = yhat[row][pred]
      end
      counter = counter +1
    end
    --compute score
    penalty = 1 - yhat[row][y[row]] + highest_nontrue
    if penalty < 0 then
      l = l + penalty
    end
  end
  return l
end

function gradients_hinge(yhat, c, x, maxindex)
  local Wprime = torch.DoubleTensor(nfeatures, nclasses)
  local grad = torch.DoubleTensor(nclasses)

  local highest_nontrue = 0
  local highest_nontrue_class = 1
  local counter = 1
  for pred=1,yhat:size()[1] do
    if counter ~= c and yhat[pred] > highest_nontrue then
      highest_nontrue = yhat[pred]
      highest_nontrue_class = pred
    end
    counter = counter +1
  end

  for i=1, nclasses do
    if yhat[c] > highest_nontrue then
      grad[i] = 0
    elseif i == highest_nontrue_class then
      grad[i] = 1
    elseif i == c then
      grad[i] = -1
    else
      grad[i] = 0
    end
  end
  
  for i=1, maxindex do
    Wprime[x[i]] = grad
  end
  return Wprime, grad
end

function gradients_logr(yhat, c, x, maxindex)
  local Wprime = torch.DoubleTensor(nfeatures, nclasses)
  local grad = torch.DoubleTensor(nclasses)
  for i=1, nclasses do
    if i == c then
      grad[i] = -1+yhat[c]
    else
      grad[i] = yhat[i]
    end
  end
  
  for i=1, maxindex do
    Wprime[x[i]] = grad
  end
  return Wprime, grad
end

function sgd(x, y, W, b, lossfunc, gradfunc)
  time = sys.clock()
  
  --iterate epochs
  for epoch=1, opt.epochs do
    print(epoch, "epoch")

    --sample minibatch, according to https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
    local itercount = 0
    for t=1, x:size()[1], opt.batchsize do
      local totaliter = x:size()[1]/opt.batchsize
      itercount = itercount + 1
      if itercount % 100 == 0 then
        print(itercount*100/totaliter, "% of Epoch done")
      end


      local inputs = torch.DoubleTensor(opt.batchsize, x:size()[2])
      local targets = torch.DoubleTensor(opt.batchsize)
      local k = 1
      for i = t,math.min(t+opt.batchsize-1,x:size()[1]) do
        inputs[k] = x[i]
        targets[k] = y[i]
        k = k+1
      end
      --it starts with 1, reduce by 1 to get size of vector
      k=k-1
      --in case the last batch is < batchsize
      if k < opt.batchsize then
        inputs = inputs:narrow(1, 1, k):clone()
        targets = targets:narrow(1, 1, k):clone()
      end
      --get the padding of the inputs
      local padding = paddings(inputs)

      --predict
      local z = torch.DoubleTensor(k, nclasses)
      for row=1, k do
        local scores = b:clone()
        for n=1, padding[row] do
          -- Scores of current row of current word index
          scores = scores + W[inputs[row][n]]
        end
        z[row] = scores
      end

      local yhat = z
      if opt.classifier == 'lr' then
        yhat = logsoftmax(z)
      end 

      --compute update
      local updateb = torch.DoubleTensor(nclasses):fill(0)
      local updatew = torch.DoubleTensor(nfeatures, nclasses):fill(0)
      for m=1, k do
        local closs = -yhat[m][targets[m]]
        gw, gb = gradfunc(yhat[m], targets[m], inputs[m], padding[m])
        updateb:add(gb/opt.batchsize)
        updatew:add(gw/opt.batchsize)
      end
      --update without regularization
      -- updatew = updatew * opt.learningrate
      -- W:add(-updatew)
      -- updateb = updateb * opt.learningrate
      -- b:add(-updateb)

      --update with regularization
      local wadd = updatew * opt.learningrate
      local badd = updateb * opt.learningrate
      local n = x:size()[1]/opt.batchsize
      local wl2 = torch.mul(W, opt.lambda * opt.learningrate / n)
      local bl2 = torch.mul(b, opt.lambda * opt.learningrate / n)
      W:add(-wadd-wl2)
      b:add(-badd-bl2)
      W:mul(1-opt.lambda * opt.learningrate / n) 
      b:mul(1-opt.lambda * opt.learningrate / n)
      W:add(-wadd)
      b:add(-badd)

    end

    --end of epoch, compute all -----------------------
    local z = torch.DoubleTensor(x:size()[1], nclasses)
    local padding = paddings(x)
    for row=1, x:size()[1] do
      local scores = b:clone()
      for n=1, padding[row] do
        -- Scores of current row of current word index
        scores = scores + W[x[row][n]]
      end
      z[row] = scores
    end

    --step 2: compute softmax (efficient)
    local yhat = logsoftmax(z)
    linearPredictor(W, b, x, y, paddings(x), true)
    --step 3: compute Loss
    local l = loss(yhat, y) + l2reg(W)
    print(l, "Loss")
    print(sys.clock()-time, "seconds so far")
  end
  return W,b
end

function comp_accuracy(y, yhat)
  y:add(-yhat:type('torch.DoubleTensor'))

  local correct_guesses = 0
  for n=1, y:size()[1] do
    if y[n] == 0 then
      correct_guesses = correct_guesses + 1
    end
  end
  local accuracy = correct_guesses/y:size()[1]
  print(accuracy, "Accuracy on Data")
  return accuracy
  -- body
end

function logsoftmax(z)
  for row=1, z:size()[1] do
    local max = torch.max(z[row])
    z[row] = z[row] - max
    local lse = torch.log(torch.sum(torch.exp(z[row])))
    z[row] = z[row] - lse + max
  end
  return z
end



function l2reg(W)
  local l2 = opt.lambda/2 * torch.norm(W,2)^2
  return l2 
end

function paddings(inputtensor)
  --get position of last non 1 in the tensor
  --can do it globally as preprocessing and store result in a tensor
  local indices = torch.Tensor(inputtensor:size()[1])
  for r=1, indices:size()[1] do
    local row = inputtensor[r]
    local i = row:size()[1]
    local found = false
    while found~=true do 
      if i < 1 then
        i=1
        --print(r, "only ones")
        break
      end
      if row[i]==1 then
        i = i - 1
      else 
        found = true
      end
    end
    indices[r] = i
  end
  return indices
end

--Main --------------------------------------------------------------

function main() 
  -- Parse input params
  opt = cmd:parse(arg)
  local f = hdf5.open(opt.datafile, 'r')
  nclasses = f:read('nclasses'):all():long()[1]
  nfeatures = f:read('nfeatures'):all():long()[1]

  print(nclasses, "classes in set")
  print(nfeatures, "features in set")

  local W = torch.DoubleTensor(nclasses, nfeatures)
  local b = torch.DoubleTensor(nclasses)
  W = W:transpose(1,2)  


  -- Read the data here
  tin = f:read('train_input'):all()
  tout = f:read('train_output'):all()

  -- Preprocessing and additional needed data
  tindices = paddings(tin)

  --subset the training data for development. 
  if opt.dev == 'true' then
    print('Development mode')
    print('Narrowing the Training Data to 100 Samples')
    tin = tin:narrow(1, 1, 100):clone()
    tout = tout:narrow(1, 1, 100):clone()
    tindices = tindices:narrow(1, 1, 100):clone()
  else
    vin = f:read('valid_input'):all()
    vout = f:read('valid_output'):all()
    vindices = paddings(vin)

    print(vin:size()[1], "validation samples")  
    print(vin:size()[2], "validation features")

    testin = f:read('test_input'):all()
    testindices = paddings(testin)
    print(testin:size()[1])
  end


  print(tin:size()[1], "Training samples")  
  print(tin:size()[2], "Training features")

  

  -- Train.
  if opt.classifier == 'lin' then
    linearPredictor(W,b, vin, vout, vindices, true)
  elseif opt.classifier == 'nb' then
    W,b = naiveBayes(W,b)
  elseif opt.classifier == 'lr' then 
    W,b = logisticReg(W,b, tin, tout, tindices)
  elseif opt.classifier == 'hinge' then 
    W, b = hinge(W,b, tin, tout, tindices)
  end

  -- Test, predictor is W,b,x,y,indices
  if opt.dev == 'test' then
    print("Training")
    linearPredictor(W, b, tin, tout, tindices, true)
    print("Validation")
    linearPredictor(W, b, vin, vout, vindices, true)
    print("Validation")
    local testout = linearPredictor(W, b, testin, vout, testindices, false)
    torch.save('test.txt', testout,'ascii')
  end



end

main()
