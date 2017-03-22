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
  local size = t:size():totable()
  table.insert(size, 1, 1)
  return t:view(unpack(size))
end

dlds.register_dataset('mpii-human-pose', function(details)
  local tmpdir = details.tmpdir

  -- local images_dir = pl.path.join('/data/dlds/cache/mpii-human-pose', 'images')
  local images_archive = details:download_file('mpii_human_pose_v1.tar.gz')
  dlds.extract_archive(tarball_file, tmpdir)
  local images_dir = pl.path.join(tmpdir, 'images')

  local annot_file = details:download_file('mpii_annot.h5')

  local annot_h5 = hdf5.open(annot_file, 'r')

  local image_names = annot_h5:read('/imgname'):all()
  local is_train = annot_h5:read('/istrain'):all()
  local centers = annot_h5:read('/center'):all()
  local scales = annot_h5:read('/scale'):all()
  local parts = annot_h5:read('/part'):all()
  local visible = annot_h5:read('/visible'):all():byte()

  local n_samples = image_names:size(1)

  local samples = {}

  for i = 1, n_samples do
    local image_name = bytes_to_string(image_names[i]:totable())
    samples[image_name] = samples[image_name] or {}

    local sample = {
      image_name = image_name,
      center = centers[i],
      scale = scales[i],
    }

    if is_train[i] == 1 then
      sample.part_coords = parts[i]
      sample.part_visible = visible[i]
      sample.subset = 'train'
    else
      sample.subset = 'test'
    end

    table.insert(samples[image_name], sample)
  end

  local train_samples = {}
  local test_samples = {}

  for k, vs in pairs(samples) do
    for i, sample in ipairs(vs) do
      if sample.subset == 'train' then
        table.insert(train_samples, sample)
      else
        table.insert(test_samples, sample)
      end
    end
  end

  print(#train_samples)
  print(#test_samples)

  local cropped = torch.ByteTensor(3, 256, 256)
  local function prepare_sample(s)
    local sz = 200 * s.scale
    local img = image.load(pl.path.join(images_dir, s.image_name), 3, 'byte')
    local scaled = image.scale(img, img:size(3) * 256 / sz, img:size(2) * 256 / sz)

    cropped:zero()
    crop(cropped, scaled, s.center[1] * 256 / sz - 127, s.center[2] * 256 / sz - 127)

    local ax0 = -(2 / sz) * s.center[1]
    local ax1 = 2 / sz
    local ay0 = -(2 / sz) * s.center[2]
    local ay1 = 2 / sz

    local m = torch.FloatTensor({
      {1/ax1, 0},
      {0, 1/ay1},
    })
    local b = torch.FloatTensor({-ax0/ax1, -ay0/ay1})

    local ps = {
      image = cropped,
      -- m and b describe the linear transform that must be applied to go from
      -- [-1, 1] normalized coordinates in the sample image to original
      -- coordinate space.
      -- orig_coords = torch.mv(m, norm_coords) + b
      m = m,
      b = b,
      subset = s.subset,
    }

    if s.subset == 'train' then
      ps.part_visible = s.part_visible

      local part_coords = torch.FloatTensor(16, 2):zero()
      for i = 1, 16 do
        if s.part_visible[i] == 1 then
          part_coords[{i, 1}] = ax1 * s.part_coords[{i, 1}] + ax0
          part_coords[{i, 2}] = ay1 * s.part_coords[{i, 2}] + ay0

          -- local x = (part_coords[{i, 1}] + 1) * 256 / 2
          -- local y = (part_coords[{i, 2}] + 1) * 256 / 2
          -- image.drawRect(cropped, x - 1, y - 1, x + 1, y + 1, {inplace = true})
        end
      end
      ps.part_coords = part_coords
    end

    return ps
  end

  local out_dir = details:make_dataset_directory()
  local out_h5 = hdf5.open(pl.path.join(out_dir, 'mpii-human-pose.h5'), 'w')

  print('Processing test set...')
  local ds_opts = {
    ['/test/images'] = hdf5.DataSetOptions():setChunked(16, 3, 256, 256),
    ['/test/transforms/m'] = hdf5.DataSetOptions():setChunked(2048, 2, 2),
    ['/test/transforms/b'] = hdf5.DataSetOptions():setChunked(2048, 2),
  }
  for i, sample_details in ipairs(test_samples) do
    local sample = prepare_sample(sample_details)

    local h5_outputs = {
      ['/test/images'] = sample.image,
      ['/test/transforms/m'] = sample.m,
      ['/test/transforms/b'] = sample.b,
    }

    for h5_path, tensor in pairs(h5_outputs) do
      out_h5[i == 1 and 'write' or 'append'](
        out_h5, h5_path, batchify(tensor), ds_opts[h5_path]
      )
    end

    xlua.progress(i, #test_samples)
  end
  print('\n')

  collectgarbage()

  print('Processing train set...')
  local ds_opts = {
    ['/train/images'] = hdf5.DataSetOptions():setChunked(16, 3, 256, 256),
    ['/train/transforms/m'] = hdf5.DataSetOptions():setChunked(2048, 2, 2),
    ['/train/transforms/b'] = hdf5.DataSetOptions():setChunked(2048, 2),
    ['/train/parts/coords'] = hdf5.DataSetOptions():setChunked(2048, 16, 2),
    ['/train/parts/visible'] = hdf5.DataSetOptions():setChunked(2048, 16),
  }
  for i, sample_details in ipairs(train_samples) do
    local sample = prepare_sample(sample_details)

    local h5_outputs = {
      ['/train/images'] = sample.image,
      ['/train/transforms/m'] = sample.m,
      ['/train/transforms/b'] = sample.b,
      ['/train/parts/coords'] = sample.part_coords,
      ['/train/parts/visible'] = sample.part_visible,
    }

    for h5_path, tensor in pairs(h5_outputs) do
      out_h5[i == 1 and 'write' or 'append'](
        out_h5, h5_path, batchify(tensor), ds_opts[h5_path]
      )
    end

    xlua.progress(i, #train_samples)
  end
  print('\n')

  out_h5:close()
end)
