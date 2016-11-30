require('torch')
local cjson = require('cjson')
local pl = require('pl.import_into')()

dlds = {
  registered_datasets = {}
}

function dlds.script_dir()
  local str = debug.getinfo(2, "S").source:sub(2)
  local dir = pl.path.abspath(str:match("(.*/)"))
  if dir:find('/%.$') then
    dir = dir:sub(1, #dir - 2)
  end

  return dir
end

local opts_string = pl.file.read(pl.path.join(dlds.script_dir(), 'config.json'))
local opts = cjson.decode(opts_string)

Details = {}
Details.__index = Details

local function check_md5sum(file, md5sum)
  local success, status, stdout =
    pl.utils.executeex('md5sum ' .. pl.utils.quote_arg(file))

  local actual_md5sum = stdout:match('^([^%s]+)')
  return md5sum == actual_md5sum, actual_md5sum
end

function Details:download_file(filename)
  local out_dir

  local url = self.resources[filename].url
  local md5sum = self.resources[filename].md5sum

  if self.opts.cache.keep_downloads then
    out_dir = pl.path.join(self.opts.cache.cache_directory, self.id)
  else
    out_dir = pl.path.join(self.tmpdir, 'downloads')
  end
  pl.dir.makepath(out_dir)

  local out_file = pl.path.join(out_dir, filename)
  if pl.path.isfile(out_file) and check_md5sum(out_file, md5sum) then
    print('Found "' .. filename .. '" in cache')
  else
    print('Downloading "' .. filename .. '"...')
    pl.utils.execute(string.format('curl -Lo %s %s',
      pl.utils.quote_arg(out_file), pl.utils.quote_arg(url)
    ))

    local md5sum_ok, actual_md5sum = check_md5sum(out_file, md5sum)
    assert(md5sum_ok, 'md5sum mismatch - got ' .. actual_md5sum .. ', expected ' .. md5sum)
  end

  return out_file
end

function Details:make_dataset_directory()
  local out_dir = pl.path.join(self.opts.datasets_directory, self.id)
  pl.dir.makepath(out_dir)
  return out_dir
end

function dlds.extract_archive(file, dest_dir)
  local format
  if file:find('%.tar%.gz$') or file:find('%.tgz$') then
    format = 'tgz'
  elseif file:find('%.tar$') then
    format = 'tar'
  elseif file:find('%.zip$') then
    format = 'zip'
  end

  print('Extracting "' .. file .. '"...')
  if format == 'tgz' then
    pl.utils.execute(string.format('tar xzf %s -C %s',
      pl.utils.quote_arg(file), pl.utils.quote_arg(dest_dir)
    ))
  elseif format == 'tar' then
    pl.utils.execute(string.format('tar xf %s -C %s',
      pl.utils.quote_arg(file), pl.utils.quote_arg(dest_dir)
    ))
  elseif format == 'zip' then
    pl.utils.execute(string.format('unzip -q %s -d %s',
      pl.utils.quote_arg(file), pl.utils.quote_arg(dest_dir)
    ))
  else
    error('unrecognised archive format')
  end
end

function dlds.gunzip(file, dest_file)
  print('Extracting "' .. file .. '"...')
  pl.utils.execute(string.format('gunzip -c %s > %s',
    pl.utils.quote_arg(file), pl.utils.quote_arg(dest_file)
  ))
end

function dlds.register_dataset(id, prepare_fn)
  local details_string = pl.file.read(pl.path.join(dlds.script_dir(), 'dlds', id, 'details.json'))
  local details = cjson.decode(details_string)
  details.opts = opts
  details.id = id
  setmetatable(details, Details)

  local tmpdir = pl.path.join(opts.temporary_directory, id)
  details.tmpdir = tmpdir

  dlds.registered_datasets[id] = {
    details = details,
    prepare_fn = prepare_fn
  }
end

function dlds.install_dataset(id)
  local registered_dataset = dlds.registered_datasets[id]
  assert(registered_dataset ~= nil, '"' .. id .. '" is not a registered dataset')
  pl.dir.makepath(registered_dataset.details.tmpdir)
  registered_dataset.prepare_fn(registered_dataset.details)
  pl.dir.rmtree(registered_dataset.details.tmpdir)
end

return dlds
