local dlds = require('dlds')
local pl = require('pl.import_into')()
local matio = require('matio')
local hdf5 = require('hdf5')

local function write_image_dataset(h5_file, path, n_examples, get_batch)
  local image_ds_opts = hdf5.DataSetOptions()
  image_ds_opts:setChunked(256, 3, 32, 32)

  local batch_size = 256
  for i = 1, n_examples, batch_size do
    local cur_batch_size = math.min(batch_size, 1 + n_examples - i)
    local sample = get_batch({i, i + cur_batch_size - 1})
    if i == 1 then
      h5_file:write(path, sample, image_ds_opts)
    else
      h5_file:append(path, sample, image_ds_opts)
    end
    xlua.progress(i, n_examples)
  end
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

  print('Processing main training examples...')
  do
    local train_mat = matio.load(train_mat_file)

    local size = train_mat.X:size()
    write_image_dataset(out_h5, '/train/main/images', size[4], function(slice)
      return train_mat.X
        :narrow(4, slice[1], (slice[2] - slice[1]) + 1)
        :permute(4, 3, 1, 2) -- HWFB => BFHW
        :contiguous()
    end)

    local train_labels = train_mat.y:byte()
    train_labels:add(train_labels:eq(0):mul(10))
    out_h5:write('/train/main/labels', train_labels, label_ds_opts)
    print('\n')
  end
  collectgarbage()

  print('Processing test examples...')
  do
    local test_mat = matio.load(test_mat_file)

    local size = test_mat.X:size()
    write_image_dataset(out_h5, '/test/images', size[4], function(slice)
      return test_mat.X
        :narrow(4, slice[1], (slice[2] - slice[1]) + 1)
        :permute(4, 3, 1, 2) -- HWFB => BFHW
        :contiguous()
    end)

    local test_labels = test_mat.y:byte()
    test_labels:add(test_labels:eq(0):mul(10))
    out_h5:write('/test/labels', test_labels, label_ds_opts)
    print('\n')
  end
  collectgarbage()

  print('Processing extra training examples...')
  do
    local extra_mat = matio.load(extra_mat_file)

    local size = extra_mat.X:size()
    write_image_dataset(out_h5, '/train/extra/images', size[4], function(slice)
      return extra_mat.X
        :narrow(4, slice[1], (slice[2] - slice[1]) + 1)
        :permute(4, 3, 1, 2) -- HWFB => BFHW
        :contiguous()
    end)

    local extra_labels = extra_mat.y:byte()
    extra_labels:add(extra_labels:eq(0):mul(10))
    out_h5:write('/train/extra/labels', extra_labels, label_ds_opts)
    print('\n')
  end
  collectgarbage()

  pl.file.write(pl.path.join(out_dir, 'classes.txt'),
    table.concat({1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, '\n') .. '\n')

  out_h5:close()
end)
