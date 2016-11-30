local dlds = require('dlds')
local pl = require('pl.import_into')()
local image = require('image')
local hdf5 = require('hdf5')
local xlua = require('xlua')

local function copy_images_to_h5(out_h5, ds_path, jpeg_files)
  local n_examples = #jpeg_files

  local image_ds_opts = hdf5.DataSetOptions()
  image_ds_opts:setChunked(4, 3, 218, 178)

  for i, jpeg_file in ipairs(jpeg_files) do
    local img = image.load(jpeg_file, 3, 'byte'):view(1, 3, 218, 178)
    if i == 1 then
      out_h5:write(ds_path, img, image_ds_opts)
    else
      out_h5:append(ds_path, img, image_ds_opts)
    end

    if i % 1000 == 0 then
      xlua.progress(i, n_examples)
    end
  end
end

dlds.register_dataset('celeba', function(details)
  local tmpdir = details.tmpdir

  local zip_file = details:download_file('img_align_celeba.zip')
  local split_file = details:download_file('list_eval_partition.txt')

  dlds.extract_archive(zip_file, tmpdir)

  local train_files = {}
  local val_files = {}
  local test_files = {}

  local file = io.open(split_file)
  local line = file:read('*line')
  repeat
    local jpeg_file, split = line:match('(%d+%.jpg)%s+(%d+)')
    jpeg_file = pl.path.join(tmpdir, 'img_align_celeba', jpeg_file)
    split = tonumber(split)

    if split == 0 then
      table.insert(train_files, jpeg_file)
    elseif split == 1 then
      table.insert(val_files, jpeg_file)
    elseif split == 2 then
      table.insert(test_files, jpeg_file)
    else
      error('unrecognized line in list_eval_partition.txt:\n' .. line)
    end

    line = file:read('*line')
  until line == nil or line == ''
  file:close()

  print(#train_files, #val_files, #test_files)

  local out_dir = details:make_dataset_directory()
  local out_h5 = hdf5.open(pl.path.join(out_dir, 'celeba.h5'), 'w')

  print('Processing training and validation examples...')
  local trainval_files = {}
  pl.tablex.insertvalues(trainval_files, train_files)
  pl.tablex.insertvalues(trainval_files, val_files)
  copy_images_to_h5(out_h5, '/train/images', trainval_files)
  print('\n')

  print('Processing test examples...')
  copy_images_to_h5(out_h5, '/test/images', test_files)
  print('\n')

  out_h5:close()
end)
