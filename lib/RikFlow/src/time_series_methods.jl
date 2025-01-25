
struct Reference_reader
    vals
    index::Array{Int64, 0}
    stds
    means
    function Reference_reader(vals)
        index = ones(Int)
        index[] = 2
        stds = std(vals, dims = 2)
        means = mean(vals, dims = 2)
        new(vals, index, stds, means)
    end
end

function get_next_item_timeseries(time_series_method::Reference_reader)
    val = time_series_method.vals[:,time_series_method.index[]]
    time_series_method.index[] += 1
    return val
end

struct MVG_sampler
    dQ_distribution
    rng
    function MVG_sampler(dQ_data, rng)
        dQ_distribution = fit(MvNormal, dQ_data .|> Float64)
        new(dQ_distribution, rng)
    end
end

function get_next_item_timeseries(time_series_method::MVG_sampler)
    return rand(time_series_method.rng, time_series_method.dQ_distribution) .|> Float32
end

struct Resampler
    vals
    rng
end

function get_next_item_timeseries(time_series_method::Resampler)
    # sample a random integer from 1 to the length of the data
    index = rand(time_series_method.rng, 1:size(time_series_method.vals, 2))
    return time_series_method.vals[:,index]
end

struct ANN
    model
    ps
    st
    scaling
    q_hist  # history of q values, newest first
    counter
    hist_var
    function ANN(file_name; q_hist = nothing)
        model, ps, st, scaling, hist_var = load_ANN(file_name)
        counter = zeros(Int)
        new(model, ps, st, scaling, q_hist, counter, hist_var)
    end
end

function get_next_item_timeseries(time_series_method::ANN, q_star)
    if !isnothing(time_series_method.q_hist)  # if the NN uses history
        if time_series_method.counter[] < size(time_series_method.q_hist, 2) # for the first few steps, directly read dQ
            time_series_method.counter[] += 1
            dQ = time_series_method.q_hist[:,end]
        else    # after that, predict dQ  (we now have enough history)
            input = vcat(q_star, time_series_method.q_hist[:]) 
            data = scale_input(input, time_series_method.scaling.in_scaling)
            pred = Lux.apply(time_series_method.model, data, time_series_method.ps, time_series_method.st)[1]
            dQ = scale_output(pred, time_series_method.scaling.out_scaling)
        end
        time_series_method.q_hist[:,2:end] = time_series_method.q_hist[:,1:end-1] # shift history
        if time_series_method.hist_var == :q
            time_series_method.q_hist[:,1] .= q_star + dQ                             # add new q to history
        elseif time_series_method.hist_var == :q_star
            time_series_method.q_hist[:,1] .= q_star
        end
    else    # if the NN does not use history, predict dQ directly from q_star
        input = q_star
        data = scale_input(input, time_series_method.scaling.in_scaling)
        pred = Lux.apply(time_series_method.model, data, time_series_method.ps, time_series_method.st)[1]
        dQ = scale_output(pred, time_series_method.scaling.out_scaling)
    end
    return dQ
end

struct LinReg
    c
    stoch_distr
    scaling
    q_hist
    spinnup_data
    counter
    hist_var
    include_predictor
    fitted_qois
    rng
    function LinReg(file_name, rng; q_hist = nothing, spinnup_data = nothing)
        c, stoch_distr, scaling, hist_var, include_predictor, fitted_qois = load(file_name, "c", "stoch_distr", "scaling", "hist_var", "include_predictor", "fitted_qois")
        scaling = scaling |> dev
        c= c |> dev
        counter = zeros(Int)
        if !isnothing(q_hist)
            @assert size(spinnup_data, 2) >= size(q_hist, 2) "Need spinnup data to fill history"
        end
        if !isnothing(spinnup_data) && isnothing(q_hist)
            @error "Spinnup not implemented without history"
        end
        new(c, stoch_distr, scaling, q_hist, spinnup_data, counter, hist_var, include_predictor, fitted_qois, rng)
    end
end

function get_next_item_timeseries(time_series_method::LinReg, q_star)
    if !isnothing(time_series_method.q_hist)  # if the model uses history
        n_qoi = size(q_star,1)
        if time_series_method.counter[] < size(time_series_method.spinnup_data,2) # for the first few steps, directly read dQ
            time_series_method.counter[] += 1
            dQ = time_series_method.spinnup_data[1:n_qoi, time_series_method.counter[]]
        else    # after that, predict dQ  (we now have enough history)
            q_star_sc = scale_input(q_star, time_series_method.scaling.in_scaling)
            if time_series_method.hist_var == :q_star_q
                q_hist_sc1 = scale_input(time_series_method.q_hist[1:n_qoi,:], time_series_method.scaling.in_scaling)
                q_hist_sc2 = scale_input(time_series_method.q_hist[n_qoi+1:end,:], time_series_method.scaling.in_scaling)
                q_hist_sc = cat(q_hist_sc1, q_hist_sc2, dims = 1)
            else
                q_hist_sc = scale_input(time_series_method.q_hist, time_series_method.scaling.in_scaling)
            end

            if time_series_method.include_predictor                
                input = vcat(q_star_sc, q_hist_sc[:])
            else
                input = q_hist_sc
            end
            
            data = vcat(input,ones(eltype(input), (1,1)))
            pred = rand(time_series_method.rng, time_series_method.stoch_distr) |> dev
            pred[time_series_method.fitted_qois,:] += time_series_method.c * data
            
            dQ = scale_output(pred, time_series_method.scaling.out_scaling)[:]
        end
        time_series_method.q_hist[:,2:end] = time_series_method.q_hist[:,1:end-1] # shift history
        if time_series_method.hist_var == :q
            time_series_method.q_hist[:,1] .= q_star + dQ                             # add new q to history
        elseif time_series_method.hist_var == :q_star
            time_series_method.q_hist[:,1] .= q_star
        elseif time_series_method.hist_var == :q_star_q
            time_series_method.q_hist[1:n_qoi,1] .= q_star + dQ
            time_series_method.q_hist[n_qoi+1:end,1] .= q_star
        end
    else    # if the model does not use history, predict dQ directly from q_star
        input = q_star
        data = vcat(scale_input(input, time_series_method.scaling.in_scaling),ones(eltype(input), (1,1)))
        pred = rand(time_series_method.rng, time_series_method.stoch_distr) |> dev
        pred[time_series_method.fitted_qois,:] += time_series_method.c * data
        dQ = scale_output(pred, time_series_method.scaling.out_scaling)[:]
    end
    return dQ
end


export get_next_item_timeseries, Reference_reader, MVG_sampler, Resampler, ANN