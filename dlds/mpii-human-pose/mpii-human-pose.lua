local dlds = require('dlds')
local pl = require('pl.import_into')()
local image = require('image')
local hdf5 = require('hdf5')

local function bytes_to_string(bytes)
  local chars = {}
  for i, byte in ipairs(bytes) do
    if byte == 0 then
      break
    end
    table.insert(chars, string.char(byte))
  end
  return table.concat(chars, '')
end

-- 1-based coords, handles out-of-bounds cases
local function crop(dst, src, x, y)
  local w = dst:size(3)
  local h = dst:size(2)
  x = torch.round(x)
  y = torch.round(y)
  local xoff = 0
  local yoff = 0
  if x < 1 then
    xoff = -x
    x = 1
  end
  if y < 1 then
    yoff = -y
    y = 1
  end
  w = math.min(w, src:size(3) - x, dst:size(3) - xoff)
  h = math.min(h, src:size(2) - y, dst:size(2) - yoff)
  if w > 0 and h > 0 then
    dst
      :narrow(3, 1 + xoff, w)
      :narrow(2, 1 + yoff, h)
      :copy(src:narrow(3, x, w):narrow(2, y, h))
  end
  return dst
end

local function batchify(t)
  if torch.type(t) == 'number' then
    return torch.FloatTensor{t}
  end
  local size = t:size():totable()
  table.insert(size, 1, 1)
  return t:view(unpack(size))
end

local cropped = torch.ByteTensor(3, 550, 550)
local function prepare_sample(s)
  local scale = s.scale
  local center = s.center:clone()

  -- Small adjustments to reduce the likelihood of cropping out joints
  center[2] = center[2] + 15 * scale
  scale = scale * 1.25

  -- We will consider this to be the bounding box size for the person
  local sz = 200 * scale
  -- The scale factor which sets the size of the "bounding box" in the output
  -- image
  local sf = 384 / sz

  -- Load the original image
  local input_image = image.load(s.image_path, 3, 'byte')

  -- Scale and crop the image to get a fixed size output image
  -- The output image is centered on the subject and scaled such that the
  -- subject has a rough "bounding box" that is 384x384 pixels in size
  local scaled = image.scale(
    input_image, input_image:size(3) * sf, input_image:size(2) * sf)
  local outh = cropped:size(2)
  local outw = cropped:size(3)
  cropped:zero()
  crop(cropped, scaled,
    center[1] * sf - math.floor(outw/2), center[2] * sf - math.floor(outh/2))

  local ps = {
    image = cropped,
    subset = s.subset,
  }

  -- Calculate matrices for transforming coordinates into processed coordinate
  -- space
  local transform_matrix = torch.FloatTensor({
    { sf , 0  },
    { 0  , sf },
  })
  local offset_matrix = torch.FloatTensor({
    { -center[1] * sf + outw / 2, -center[2] * sf + outh / 2 },
  })

  if s.subset ~= 'test' then
    local part_coords = torch.mm(s.part_coords:float(), transform_matrix)
    part_coords:add(offset_matrix:expandAs(part_coords))

    local part_visible = s.part_visible

    ps.part_visible = part_visible
    ps.part_coords = part_coords

    -- -- Draw joints over the image. Only useful for debugging
    -- for i = 1, 16 do
    --   if part_visible[i] == 1 then
    --     local x = part_coords[{i, 1}]
    --     local y = part_coords[{i, 2}]
    --     image.drawRect(cropped, x - 1, y - 1, x + 1, y + 1, {inplace = true, color = {255, 0, 0}})
    --   end
    -- end
  end

  -- Calculate matrices for transforming coords back into original image space
  local inv_matrix = torch.inverse(transform_matrix:double())
  ps.transform_m = inv_matrix:float()
  ps.transform_b = -torch.mm(offset_matrix:double(), inv_matrix):float()

  return ps
end

local function process_subset(out_h5, subset, ids, annotations)
  local ds_opts = {
    ['images'] = hdf5.DataSetOptions():setChunked(4, 3, 256, 256),
    ['transforms/m'] = hdf5.DataSetOptions():setChunked(2048, 2, 2),
    ['transforms/b'] = hdf5.DataSetOptions():setChunked(2048, 1, 2),
    ['parts/coords'] = hdf5.DataSetOptions():setChunked(2048, 16, 2),
    ['parts/visible'] = hdf5.DataSetOptions():setChunked(2048, 16),
    ['parts/visible'] = hdf5.DataSetOptions():setChunked(2048, 16),
    ['imgnames'] = hdf5.DataSetOptions():setChunked(2048, 16),
    ['normalize'] = hdf5.DataSetOptions():setChunked(2048),
  }

  for i, id in ipairs(ids) do
    local sample = {
      subset = subset,
      image_path = annotations.image_paths[id],
      imgname = annotations.imgnames[id],
      center = annotations.centers[id],
      scale = annotations.scales[id],
      part_coords = annotations.parts[id],
      part_visible = annotations.visible[id],
    }

    local prepared_sample = prepare_sample(sample)

    local h5_outputs = {
      ['images'] = prepared_sample.image,
      ['transforms/m'] = prepared_sample.transform_m,
      ['transforms/b'] = prepared_sample.transform_b,
      ['imgnames'] = sample.imgname,
      ['normalize'] = annotations.normalize[id],
    }

    if subset ~= 'test' then
      h5_outputs['parts/coords'] = prepared_sample.part_coords
      h5_outputs['parts/visible'] = prepared_sample.part_visible
    end

    for h5_path, tensor in pairs(h5_outputs) do
      out_h5[i == 1 and 'write' or 'append'](
        out_h5, '/' .. subset .. '/' .. h5_path, batchify(tensor), ds_opts[h5_path]
      )
    end

    xlua.progress(i, #ids)
  end

  print('\n')
end

dlds.register_dataset('mpii-human-pose', function(details)
  local tmpdir = details.tmpdir

  -- local images_dir = pl.path.join('/data/dlds/cache/mpii-human-pose', 'images')
  local images_archive = details:download_file('mpii_human_pose_v1.tar.gz')
  dlds.extract_archive(images_archive, tmpdir)
  local images_dir = pl.path.join(tmpdir, 'images')

  local all_annot_file = details:download_file('mpii_annot_all.h5')
  local val_annot_file = details:download_file('mpii_annot_valid.h5')

  local annot_h5 = hdf5.open(all_annot_file, 'r')
  local image_indices = annot_h5:read('/index'):all():long()
  local is_train = annot_h5:read('/istrain'):all()
  local imgnames = annot_h5:read('/imgname'):all()
  local people = annot_h5:read('/person'):all():long()
  local image_paths = {}
  for i = 1, imgnames:size(1) do
    local image_name = bytes_to_string(imgnames[i]:totable())
    image_paths[i] = pl.path.join(images_dir, image_name)
  end
  local annotations = {
    centers = annot_h5:read('/center'):all(),
    scales = annot_h5:read('/scale'):all(),
    parts = annot_h5:read('/part'):all(),
    visible = annot_h5:read('/visible'):all():byte(),
    normalize = annot_h5:read('/normalize'):all(),
    imgnames = imgnames,
    image_paths = image_paths,
  }
  annot_h5:close()

  local val_annot_h5 = hdf5.open(val_annot_file, 'r')
  local val_image_indices = val_annot_h5:read('/index'):all():long()
  local val_people = val_annot_h5:read('/person'):all():long()
  val_annot_h5:close()

  local train_ids = {}
  local val_ids = {}
  local test_ids = {}

  -- An "identifier" is the image index concatenated with the index of the
  -- person in the image
  local all_identifiers = image_indices:cat(people, 2)
  local val_identifiers = val_image_indices:cat(val_people, 2)

  local val_pos = 1
  for i = 1, image_indices:size(1) do
    if val_pos <= val_image_indices:size(1) and all_identifiers[i]:equal(val_identifiers[val_pos]) then
      table.insert(val_ids, i)
      val_pos = val_pos + 1
    elseif is_train[i] == 0 then
      table.insert(test_ids, i)
    else
      table.insert(train_ids, i)
    end
  end

  print(#train_ids, #val_ids, #test_ids)

  local out_dir = details:make_dataset_directory()
  local out_h5 = hdf5.open(pl.path.join(out_dir, 'mpii-human-pose.h5'), 'w')

  print('Processing train set...')
  process_subset(out_h5, 'train', train_ids, annotations)
  collectgarbage()

  print('Processing validation set...')
  process_subset(out_h5, 'val', val_ids, annotations)
  collectgarbage()

  print('Processing test set...')
  process_subset(out_h5, 'test', test_ids, annotations)
  collectgarbage()

  out_h5:close()
end)
