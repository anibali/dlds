local dlds = require('dlds')
local pl = require('pl.import_into')()
local matio = require('matio')
local hdf5 = require('hdf5')
local xlua = require('xlua')

local function write_image_dataset(h5_file, path, n_examples, get_batch)
  local image_ds_opts = hdf5.DataSetOptions()
  image_ds_opts:setChunked(16, 3, 96, 96)

  local batch_size = 64
  for i = 1, n_examples, batch_size do
    local cur_batch_size = math.min(batch_size, 1 + n_examples - i)
    local sample = get_batch({i, i + cur_batch_size - 1})
    sample = sample
      :transpose(1, 2) -- pixels x batch => batch x pixels
      :reshape(cur_batch_size, 3, 96, 96) -- batch x pixels => BFWH
      :transpose(3, 4) -- BFWH => BFHW
    if i == 1 then
      h5_file:write(path, sample, image_ds_opts)
    else
      h5_file:append(path, sample, image_ds_opts)
    end
    xlua.progress(i, n_examples)
  end
end

dlds.register_dataset('stl-10', function(details)
  local tmpdir = details.tmpdir

  local tarball_file = details:download_file('stl10_matlab.tar.gz')
  dlds.extract_archive(tarball_file, tmpdir)

  local test_mat_file = pl.path.join(tmpdir, 'stl10_matlab', 'test.mat')
  local train_mat_file = pl.path.join(tmpdir, 'stl10_matlab', 'train.mat')
  local unlabeled_mat_file = pl.path.join(tmpdir, 'stl10_matlab', 'unlabeled.mat')

  local out_dir = details:make_dataset_directory()
  local out_h5 = hdf5.open(pl.path.join(out_dir, 'stl-10.h5'), 'w')

  local label_ds_opts = hdf5.DataSetOptions()
  label_ds_opts:setChunked(4096, 1)

  matio.use_lua_strings = true

  print('Processing unlabeled training examples...')
  local unlabeled_h5 = hdf5.open(unlabeled_mat_file, 'r')
  local data = unlabeled_h5:read('/X')
  local size = data:dataspaceSize()
  write_image_dataset(out_h5, '/train/unlabeled/images', size[2], function(slice)
    return data:partial({1, size[1]}, slice):byte()
  end)
  unlabeled_h5:close()

  print('Processing labeled training examples...')
  local train_mat = matio.load(train_mat_file)

  local fold_ds_opts = hdf5.DataSetOptions()
  fold_ds_opts:setChunked(1024, 1)
  local fold_indices = torch.LongTensor(10, 1000)
  for i = 1, 10 do
    fold_indices[i]:copy(train_mat.fold_indices[i]:squeeze())
  end
  out_h5:write('/train/labeled/fold_indices', fold_indices, fold_ds_opts)

  write_image_dataset(out_h5, '/train/labeled/images', train_mat.X:size(1), function(slice)
    return train_mat.X:narrow(1, slice[1], 1 + slice[2] - slice[1]):byte()
  end)
  out_h5:write('/train/labeled/labels', train_mat.y:byte(), label_ds_opts)

  print('Processing test examples...')
  local test_mat = matio.load(test_mat_file)

  write_image_dataset(out_h5, '/test/images', test_mat.X:size(1), function(slice)
    return test_mat.X:narrow(1, slice[1], 1 + slice[2] - slice[1]):byte()
  end)
  out_h5:write('/test/labels', test_mat.y:byte(), label_ds_opts)

  pl.file.write(pl.path.join(out_dir, 'classes.txt'),
    table.concat(test_mat.class_names, '\n') .. '\n')

  out_h5:close()
end)
