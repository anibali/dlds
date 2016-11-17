local dlds = require('dlds')
local pl = require('pl.import_into')()
local matio = require('matio')
local hdf5 = require('hdf5')

dlds.register_dataset('cifar-100', function(details)
  local tmpdir = details.tmpdir

  local tarball_file = details:download_file('cifar-100-matlab.tar.gz')
  dlds.extract_archive(tarball_file, tmpdir)

  matio.use_lua_strings = true
  local function load_mat_file(short_name)
    local mat_file = pl.path.join(tmpdir, 'cifar-100-matlab', short_name .. '.mat')
    local mat = matio.load(mat_file)
    return mat
  end

  local out_dir = details:make_dataset_directory()
  local out_h5 = hdf5.open(pl.path.join(out_dir, 'cifar-100.h5'), 'w')

  local label_ds_opts = hdf5.DataSetOptions()
  label_ds_opts:setChunked(4096, 1)
  local image_ds_opts = hdf5.DataSetOptions()
  image_ds_opts:setChunked(256, 3, 32, 32)

  print('Processing training set...')
  do
    local train_mat = load_mat_file('train')

    local images = train_mat.data:reshape(train_mat.data:size(1), 3, 32, 32)
    local labels = train_mat.fine_labels:add(1)

    out_h5:write('/train/images', images, image_ds_opts)
    out_h5:write('/train/labels', labels, label_ds_opts)
  end
  collectgarbage()

  local superclass_associations = torch.ByteTensor(100, 1)

  print('Processing test set...')
  do
    local test_mat = load_mat_file('test')

    local images = test_mat.data:reshape(test_mat.data:size(1), 3, 32, 32)
    local labels = test_mat.fine_labels:add(1)

    out_h5:write('/test/images', images, image_ds_opts)
    out_h5:write('/test/labels', labels, label_ds_opts)

    for i = 1, labels:size(1) do
      superclass_associations[labels[{i, 1}]] = test_mat.coarse_labels[{i, 1}] + 1
    end
  end
  collectgarbage()

  out_h5:write('/meta/superclasses', superclass_associations, label_ds_opts)

  local meta_mat = load_mat_file('meta')

  pl.file.write(pl.path.join(out_dir, 'classes.txt'),
    table.concat(meta_mat.fine_label_names, '\n') .. '\n')

  pl.file.write(pl.path.join(out_dir, 'superclasses.txt'),
    table.concat(meta_mat.coarse_label_names, '\n') .. '\n')

  out_h5:close()
end)
