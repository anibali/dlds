local dlds = require('dlds')
local pl = require('pl.import_into')()
local hdf5 = require('hdf5')
local matio = require('matio')
local image = require('image')

local function process_subset(out_h5, subset, annots, img_dir)
  local gen = torch.Generator()
  torch.manualSeed(gen, 12345)
  local indices = torch.randperm(gen, #annots):totable()

  local ds_opts = {
    ['images'] = hdf5.DataSetOptions():setChunked(1, 3, 500, 500),
    ['dims'] = hdf5.DataSetOptions():setChunked(2048, 2),
    ['labels'] = hdf5.DataSetOptions():setChunked(2048, 1),
    ['boxes'] = hdf5.DataSetOptions():setChunked(2048, 4),
    ['imgnames'] = hdf5.DataSetOptions():setChunked(2048, 16),
  }

  local imgname_tensor = torch.CharTensor(1, 16)
  local padded_img = torch.ByteTensor(1, 3, 500, 500)
  local dim = torch.FloatTensor(1, 2)

  for i, index in ipairs(indices) do
    local annot = annots[index]

    local chars = torch.CharTensor(torch.CharStorage():string(annot.imgname))
    imgname_tensor:zero()
    imgname_tensor[{1, {1, chars:size(1)}}]:copy(chars)

    local img = image.load(pl.path.join(img_dir, annot.imgname), 3, 'byte')

    local scale_factor = 1
    local width = img:size(3)
    local height = img:size(2)
    local max_dim = math.max(width, height)
    if max_dim > 500 then
      scale_factor = 500 / max_dim
      width = width * scale_factor
      height = height * scale_factor
      img = image.scale(img, width, height)
    end

    dim[{1, 1}] = width
    dim[{1, 2}] = height

    padded_img:zero()
    padded_img[1]:sub(1, 3, 1, height, 1, width):copy(img)

    local h5_outputs = {
      ['images'] = padded_img,
      ['dims'] = dim,
      ['labels'] = annot.label,
      ['boxes'] = annot.box * scale_factor,
      ['imgnames'] = imgname_tensor,
    }

    for h5_path, tensor in pairs(h5_outputs) do
      out_h5[i == 1 and 'write' or 'append'](
        out_h5, '/' .. subset .. '/' .. h5_path, tensor, ds_opts[h5_path]
      )
    end

    xlua.progress(i, #indices)
  end

  print('\n')
end

dlds.register_dataset('stanford-cars', function(details)
  local tmpdir = details.tmpdir

  local annots_file = details:download_file('cars_annos.mat')

  matio.use_lua_strings = true

  local annots_mat = matio.load(annots_file)
  local classes = annots_mat.class_names
  local annotations = annots_mat.annotations[1]

  local images_file = details:download_file('car_ims.tgz')
  dlds.extract_archive(images_file, tmpdir)
  local img_dir = pl.path.join(tmpdir, 'car_ims')

  local train_annots = {}
  local test_annots = {}
  for i, annot in ipairs(annotations) do
    local is_test = annot.test:sum() == 1

    local new_annot = {
      box = torch.cat(annot.bbox_x1, annot.bbox_y1)
        :cat(annot.bbox_x2):cat(annot.bbox_y2):float(),
      label = annot.class:long(),
      imgname = annot.relative_im_path:match('car_ims/(%d+%.jpg)'),
    }

    if is_test then
      table.insert(test_annots, new_annot)
    else
      table.insert(train_annots, new_annot)
    end
  end

  local out_dir = details:make_dataset_directory()
  local out_h5 = hdf5.open(pl.path.join(out_dir, 'stanford-cars.h5'), 'w')

  print('Processing training set...')
  process_subset(out_h5, 'train', train_annots, img_dir)

  print('Processing test set...')
  process_subset(out_h5, 'test', test_annots, img_dir)

  pl.file.write(pl.path.join(out_dir, 'classes.txt'),
  table.concat(classes, '\n') .. '\n')

  out_h5:close()
end)
