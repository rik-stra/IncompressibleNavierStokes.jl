
struct Reference_reader
    vals
    index::Array{Int64, 0}
    function Reference_reader(vals)
        index = ones(Int)
        index[] = 2
        new(vals, index)
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
        dQ_distribution = fit(MvNormal, dQ_data)
        new(dQ_distribution, rng)
    end
end

function get_next_item_timeseries(time_series_method::MVG_sampler)
    return rand(time_series_method.rng, time_series_method.dQ_distribution)
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


export get_next_item_timeseries, Reference_reader, MVG_sampler, Resampler, ANN