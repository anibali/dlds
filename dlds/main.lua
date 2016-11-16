-- This is an executable script which allows the user to install datasets from
-- the command line.

local dlds = require('dlds')
local pl = require('pl.import_into')()

local dataset_id = arg[1]

local dataset_script = pl.path.join(dlds.script_dir(), dataset_id, dataset_id .. '.lua')
assert(pl.path.isfile(dataset_script),
  'no dataset script found for "' .. dataset_id .. '"')

dofile(dataset_script)
dlds.install_dataset(dataset_id)
