local dlds = require('dlds')
local pl = require('pl.import_into')()
local hdf5 = require('hdf5')

local function pluck_int32(bytes, index)
  return
    (bytes[index + 0] * 0x01000000) +
    (bytes[index + 1] * 0x00010000) +
    (bytes[index + 2] * 0x00000100) +
    (bytes[index + 3] * 0x00000001)
end

local function read_images(gz_file, tmpdir)
  local images_file = pl.path.join(tmpdir, 'extracted-' .. gz_file:gsub('%W', ''))
  dlds.gunzip(gz_file, images_file)

  local images_bytes = torch.ByteStorage(images_file)
  assert(pluck_int32(images_bytes, 1) == 0x00000803, 'magic number mismatch')

  local n_images = pluck_int32(images_bytes, 5)
  local n_rows = pluck_int32(images_bytes, 9)
  local n_cols = pluck_int32(images_bytes, 13)

  local images = torch.ByteTensor(images_bytes, 16 + 1,
    torch.LongStorage{n_images, 1, n_rows, n_cols})

  return images
end

local function read_labels(gz_file, tmpdir)
  local labels_file = pl.path.join(tmpdir, 'extracted-' .. gz_file:gsub('%W', ''))
  dlds.gunzip(gz_file, labels_file)

  local labels_bytes = torch.ByteStorage(labels_file)
  assert(pluck_int32(labels_bytes, 1) == 0x00000801, 'magic number mismatch')

  local n_labels = pluck_int32(labels_bytes, 5)

  local labels = torch.ByteTensor(labels_bytes, 8 + 1, torch.LongStorage{n_labels, 1})

  -- Replace label index 0 with 10
  labels:add(labels:eq(0):mul(10))

  return labels
end

dlds.register_dataset('mnist', function(details)
  local tmpdir = details.tmpdir

  local train_images_gz = details:download_file('train-images-idx3-ubyte.gz')
  local train_labels_gz = details:download_file('train-labels-idx1-ubyte.gz')
  local test_images_gz = details:download_file('t10k-images-idx3-ubyte.gz')
  local test_labels_gz = details:download_file('t10k-labels-idx1-ubyte.gz')

  local out_dir = details:make_dataset_directory()
  local out_h5 = hdf5.open(pl.path.join(out_dir, 'mnist.h5'), 'w')

  local label_ds_opts = hdf5.DataSetOptions()
  label_ds_opts:setChunked(4096, 1)
  local image_ds_opts = hdf5.DataSetOptions()
  image_ds_opts:setChunked(512, 1, 28, 28)

  do
    local train_images = read_images(train_images_gz, tmpdir)
    out_h5:write('/train/images', train_images, image_ds_opts)

    local train_labels = read_labels(train_labels_gz, tmpdir)
    out_h5:write('/train/labels', train_labels, label_ds_opts)
  end
  collectgarbage()

  do
    local test_images = read_images(test_images_gz, tmpdir)
    out_h5:write('/test/images', test_images, image_ds_opts)

    local test_labels = read_labels(test_labels_gz, tmpdir)
    out_h5:write('/test/labels', test_labels, label_ds_opts)
  end
  collectgarbage()

  pl.file.write(pl.path.join(out_dir, 'classes.txt'),
    table.concat({1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, '\n') .. '\n')

  out_h5:close()
end)
