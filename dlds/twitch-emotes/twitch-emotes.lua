local dlds = require('dlds')
local pl = require('pl.import_into')()
local cjson = require('cjson')
local hdf5 = require('hdf5')
local xlua = require('xlua')
local gm = require('graphicsmagick')

local function are_tables_equal(t1, t2)
  return pl.tablex.compare(t1, t2, function(a, b) return a == b end)
end

local function load_image(filename)
  return gm.Image(filename)
    :setBackground(1, 1, 1)
    :flatten()
    :toTensor('byte', 'RGB', 'DHW')
end

dlds.register_dataset('twitch-emotes', function(details)
  local tmpdir = details.tmpdir

  local json_file = details:download_file('subscriber_20170216.json')
  local file = io.open(json_file, 'r')
  local json_string = file:read('*a')
  file:close()

  local json = cjson.decode(json_string)

  local url_template = json.template.small

  local emotes = {}

  local max_code_len = 0
  for channel_name, channel in pairs(json.channels) do
    for i, emote in ipairs(channel.emotes) do
      table.insert(emotes, {
        code = emote.code,
        image_url = url_template:gsub('{image_id}', emote.image_id)
      })
      if #emote.code > max_code_len then
        max_code_len = #emote.code
      end
    end
  end
  assert(max_code_len < 32, 'max emote code length exceeded')

  local out_dir = details:make_dataset_directory()
  local out_h5 = hdf5.open(pl.path.join(out_dir, 'twitch-emotes.h5'), 'w')

  local image_ds_opts = hdf5.DataSetOptions()
  image_ds_opts:setChunked(512, 1, 28, 28)

  local code_ds_opts = hdf5.DataSetOptions()
  code_ds_opts:setChunked(1024, 32)

  local train_images = torch.ByteTensor(#emotes, 3, 28, 28)
  local train_codes = torch.CharTensor(#emotes, 32):zero()
  local pos = 1

  for i, emote in ipairs(emotes) do
    local image_file = details:download_tmp_file(emotes[i].image_url, 'emote.png')
    local ok, img = pcall(load_image, image_file)
    if ok and are_tables_equal(img:size():totable(), {3, 28, 28}) then
      train_images[pos]:copy(img)
      for j = 1, #emote.code do
        train_codes[{pos, j}] = emote.code:byte(j)
      end
      pos = pos + 1
    else
      print(('Skipping %s'):format(emote.code))
    end
    xlua.progress(i, #emotes)

    if i % 100 == 0 then
      collectgarbage()
    end
  end

  train_images = train_images:narrow(1, 1, pos - 1)
  train_codes = train_codes:narrow(1, 1, pos - 1)

  out_h5:write('/train/images', train_images, image_ds_opts)
  out_h5:write('/train/codes', train_codes, code_ds_opts)

  out_h5:close()
end)
