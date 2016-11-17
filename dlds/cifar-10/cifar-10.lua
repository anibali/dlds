local dlds = require('dlds')
local pl = require('pl.import_into')()
local matio = require('matio')
local hdf5 = require('hdf5')

dlds.register_dataset('cifar-10', function(details)
  local tmpdir = details.tmpdir

  local tarball_file = details:download_file('cifar-10-matlab.tar.gz')
  dlds.extract_archive(tarball_file, tmpdir)

  matio.use_lua_strings = true
  local function load_mat_file(short_name)
    local mat_file = pl.path.join(tmpdir, 'cifar-10-batches-mat', short_name .. '.mat')
    local mat = matio.load(mat_file)
    return mat
  end

  local out_dir = details:make_dataset_directory()
  local out_h5 = hdf5.open(pl.path.join(out_dir, 'cifar-10.h5'), 'w')

  local label_ds_opts = hdf5.DataSetOptions()
  label_ds_opts:setChunked(4096, 1)
  local image_ds_opts = hdf5.DataSetOptions()
  image_ds_opts:setChunked(256, 3, 32, 32)

  for i = 1, 5 do
    print('Processing training batch ' .. i .. ' of 5...')

    local data_batch_mat = load_mat_file('data_batch_' .. i)

    local batch_images = data_batch_mat.data:reshape(data_batch_mat.data:size(1), 3, 32, 32)
    local batch_labels = data_batch_mat.labels:add(1)

    if i == 1 then
      out_h5:write('/train/images', batch_images, image_ds_opts)
      out_h5:write('/train/labels', batch_labels, label_ds_opts)
    else
      out_h5:append('/train/images', batch_images, image_ds_opts)
      out_h5:append('/train/labels', batch_labels, label_ds_opts)
    end
  end

  print('Processing test batch...')
  local test_batch_mat = load_mat_file('test_batch')

  local batch_images = test_batch_mat.data:reshape(test_batch_mat.data:size(1), 3, 32, 32)
  local batch_labels = test_batch_mat.labels:add(1)

  out_h5:write('/test/images', batch_images, image_ds_opts)
  out_h5:write('/test/labels', batch_labels, label_ds_opts)

  local batches_meta_mat = load_mat_file('batches.meta')

  pl.file.write(pl.path.join(out_dir, 'classes.txt'),
    table.concat(batches_meta_mat.label_names, '\n') .. '\n')

  out_h5:close()
end)
