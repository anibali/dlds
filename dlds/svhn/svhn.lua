local dlds = require('dlds')
local pl = require('pl.import_into')()
local matio = require('matio')
local hdf5 = require('hdf5')

local function write_image_dataset(h5_file, path, data_mat, gen)
  local image_ds_opts = hdf5.DataSetOptions()
  image_ds_opts:setChunked(256, 3, 32, 32)
  local label_ds_opts = hdf5.DataSetOptions()
  label_ds_opts:setChunked(4096, 1)

  local n_examples = data_mat.X:size(4)
  local indices = torch.randperm(gen, n_examples):long()
  local batch_size = 256

  for i = 1, n_examples, batch_size do
    local cur_batch_size = math.min(batch_size, 1 + n_examples - i)
    local sample = data_mat.X
      :index(4, indices:narrow(1, i, cur_batch_size))
      :permute(4, 3, 1, 2) -- HWFB => BFHW
      :contiguous()
    if i == 1 then
      h5_file:write(path .. '/images', sample, image_ds_opts)
    else
      h5_file:append(path .. '/images', sample, image_ds_opts)
    end
    xlua.progress(i, n_examples)
  end

  local labels = data_mat.y:byte():index(1, indices)
  labels:add(labels:eq(0):mul(10))
  h5_file:write(path .. '/labels', labels, label_ds_opts)
end

dlds.register_dataset('svhn', function(details)
  local tmpdir = details.tmpdir

  local train_mat_file = details:download_file('train_32x32.mat')
  local test_mat_file = details:download_file('test_32x32.mat')
  local extra_mat_file = details:download_file('extra_32x32.mat')

  local out_dir = details:make_dataset_directory()
  local out_h5 = hdf5.open(pl.path.join(out_dir, 'svhn.h5'), 'w')

  local label_ds_opts = hdf5.DataSetOptions()
  label_ds_opts:setChunked(4096, 1)

  local gen = torch.Generator()
  torch.manualSeed(gen, 12345)

  print('Processing main training examples...')
  do
    local train_mat = matio.load(train_mat_file)
    write_image_dataset(out_h5, '/train/main', train_mat, gen)
    print('\n')
  end
  collectgarbage()

  print('Processing test examples...')
  do
    local test_mat = matio.load(test_mat_file)
    write_image_dataset(out_h5, '/test', test_mat, gen)
    print('\n')
  end
  collectgarbage()

  print('Processing extra training examples...')
  do
    local extra_mat = matio.load(extra_mat_file)
    write_image_dataset(out_h5, '/train/extra', extra_mat, gen)
    print('\n')
  end
  collectgarbage()

  pl.file.write(pl.path.join(out_dir, 'classes.txt'),
    table.concat({1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, '\n') .. '\n')

  out_h5:close()
end)
