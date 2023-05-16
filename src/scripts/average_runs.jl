using TensorBoardLogger, ValueHistories, Interpolations, Statistics, Logging

function retrieve_logged_data(log_name::String, index::Int)
   tb_reader = TBReader(log_name)
   hist = convert(MVHistory, tb_reader)
   hist_keys = []
   for key in keys(hist)
      push!(hist_keys, key)
   end
   t, y = get(hist, Symbol("reward/total_reward_per_episode.rewards[end]"))
end

function run_averaging_internal(; directory, pattern)
   files = readdir(directory)
   logs = filter(file -> occursin(pattern, file), files)
   interp_runs = []

   for log in logs
      t, y = retrieve_logged_data(string(directory, log), 11) # 6 is the reward key
      push!(interp_runs, LinearInterpolation(t, y))
   end

   common_timepoints = range(200, stop=30000, length=300)
   values_run_common = zeros(300, length(interp_runs))

   for run_index in 4:length(interp_runs)
      values_run_common[:, run_index] = interp_runs[run_index](common_timepoints)
   end

   average_run = mean(values_run_common, dims=2)
end

function run_averaging(;directory="newer_logs/", environment::String, select_baseline::Bool)
   if select_baseline
      pattern_str = "^" * environment * "(?=.*baseline)(?!.*groundtruth).*"
  else
      pattern_str = "^" * environment * "(?!.*baseline|.*groundtruth).*"
  end
  pattern = Regex(pattern_str)
  run_averaging_internal(directory=directory, pattern=pattern)
end

average_reward = run_averaging(environment="Door", select_baseline=false)

lg = TBLogger("newer_logs/average_door_repr")
with_logger(lg) do
   for reward in average_reward
      @info  "reward" reward
   end
end