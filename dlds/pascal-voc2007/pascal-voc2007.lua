local dlds = require('dlds')
local pl = require('pl.import_into')()
local hdf5 = require('hdf5')
local lom = require('lxp.lom')
local xpath = require('luaxpath')
local image = require('image')

local function read_lines(file_path)
  local file = io.open(file_path)
  local t = {}
  local line
  repeat
    line = file:read('*line')
    table.insert(t, line)
  until line == nil
  file:close()

  return t
end

local function make_ds_opts(...)
  local ds_opts = hdf5.DataSetOptions()
  ds_opts:setChunked(...)
  return ds_opts
end

dlds.register_dataset('pascal-voc2007', function(details)
  local tmpdir = details.tmpdir

  local trainval_tar = details:download_file('VOCtrainval_06-Nov-2007.tar')
  local test_tar = details:download_file('VOCtest_06-Nov-2007.tar')

  dlds.extract_archive(trainval_tar, tmpdir)
  dlds.extract_archive(test_tar, tmpdir)

  local classes = read_lines(pl.path.join(dlds.script_dir(), 'classes.txt'))
  local classes_inv = {}
  for i, v in ipairs(classes) do
    classes_inv[v] = i
  end

  local out_dir = details:make_dataset_directory()
  local out_h5 = hdf5.open(pl.path.join(out_dir, 'pascal-voc2007.h5'), 'w')

  local image_ds_opts = hdf5.DataSetOptions()
  image_ds_opts:setChunked(1, 3, 500, 500)

  local padded_img = torch.ByteTensor(1, 3, 500, 500)

  local images_dir = pl.path.join(tmpdir, 'VOCdevkit', 'VOC2007', 'JPEGImages')

  local max_objs = 50

  local function process_examples(ids, subset)
    local tensor_boxes = torch.LongTensor(#ids, max_objs, 4):zero()
    local tensor_dims = torch.LongTensor(#ids, 2):zero()
    local tensor_objcount = torch.ByteTensor(#ids, 1):zero()
    local tensor_labels = torch.ByteTensor(#ids, max_objs, 1):zero()
    local tensor_difficult = torch.ByteTensor(#ids, max_objs, 1):zero()
    local tensor_truncated = torch.ByteTensor(#ids, max_objs, 1):zero()

    for i, id in ipairs(ids) do
      local xml_doc = lom.parse(pl.file.read(pl.path.join(
        tmpdir, 'VOCdevkit', 'VOC2007', 'Annotations', id .. '.xml')))

      local image_file = pl.path.join(images_dir, id .. '.jpg')
      local img = image.load(image_file, 3, 'byte')

      tensor_dims[{i, 1}] = img:size(2)
      tensor_dims[{i, 2}] = img:size(3)

      padded_img:zero()
      padded_img[1]:narrow(2, 1, img:size(2)):narrow(3, 1, img:size(3)):copy(img)

      if i == 1 then
        out_h5:write('/' .. subset .. '/images', padded_img, image_ds_opts)
      else
        out_h5:append('/' .. subset .. '/images', padded_img, image_ds_opts)
      end

      local obj_nodes = xpath.selectNodes(xml_doc, '/annotation/object')
      tensor_objcount[{i, 1}] = #obj_nodes

      for obj_index, obj_node in ipairs(obj_nodes) do
        local name = xpath.selectNodes(obj_node, '/object/name/text()')[1]
        local label = classes_inv[name]
        assert(label ~= nil, 'unrecognised class name: ' .. name)
        local x_min = tonumber(xpath.selectNodes(obj_node, '/object/bndbox/xmin/text()')[1])
        local y_min = tonumber(xpath.selectNodes(obj_node, '/object/bndbox/ymin/text()')[1])
        local x_max = tonumber(xpath.selectNodes(obj_node, '/object/bndbox/xmax/text()')[1])
        local y_max = tonumber(xpath.selectNodes(obj_node, '/object/bndbox/ymax/text()')[1])
        local truncated = tonumber(xpath.selectNodes(obj_node, '/object/truncated/text()')[1])
        local difficult = tonumber(xpath.selectNodes(obj_node, '/object/difficult/text()')[1])

        tensor_labels[{i, obj_index, 1}] = label
        tensor_difficult[{i, obj_index, 1}] = difficult
        tensor_truncated[{i, obj_index, 1}] = truncated

        tensor_boxes[{i, obj_index, 1}] = x_min
        tensor_boxes[{i, obj_index, 2}] = y_min
        tensor_boxes[{i, obj_index, 3}] = x_max
        tensor_boxes[{i, obj_index, 4}] = y_max
      end
    end

    out_h5:write('/' .. subset .. '/dims', tensor_dims, make_ds_opts(1024, 2))
    out_h5:write('/' .. subset .. '/objcount', tensor_objcount, make_ds_opts(1024, 1))
    out_h5:write('/' .. subset .. '/labels', tensor_labels, make_ds_opts(1024, 50, 1))
    out_h5:write('/' .. subset .. '/boxes', tensor_boxes, make_ds_opts(1024, 50, 4))
    out_h5:write('/' .. subset .. '/difficult', tensor_difficult, make_ds_opts(1024, 50, 1))
    out_h5:write('/' .. subset .. '/truncated', tensor_truncated, make_ds_opts(1024, 50, 1))
  end

  local train_ids = read_lines(pl.path.join(
    tmpdir, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'train.txt'))

  local val_ids = read_lines(pl.path.join(
    tmpdir, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'val.txt'))

  local trainval_ids = {}
  pl.tablex.insertvalues(trainval_ids, train_ids)
  pl.tablex.insertvalues(trainval_ids, val_ids)

  print('Processing training set...')
  process_examples(trainval_ids, 'train')

  local test_ids = read_lines(pl.path.join(
    tmpdir, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'test.txt'))

  print('Processing test set...')
  process_examples(test_ids, 'test')

  pl.file.write(pl.path.join(out_dir, 'classes.txt'),
    table.concat(classes, '\n') .. '\n')

  out_h5:close()
end)
