"""
Reorder matrices rows and columns to move high values close to the diagonal.

!!! note

    Instead of providing the limited `oclust` (like the R version), this includes [`ehclust`](@ref), an extended version
    of [hclust]*https://github.com/JuliaStats/Clustering.jl/blob/master/src/hclust.jl).
"""
module Slanter

using Reexport
include("enhanced_hclust.jl")
@reexport using .EnhancedHclust

export reorder_hclust
export slanted_orders
export slanted_reorder

using Clustering
using LinearAlgebra
using Statistics
using StatsBase

"""
    function slanted_orders(
        data::AbstractMatrix{<:Real};
        order_rows::Bool=true,
        order_cols::Bool=true,
        squared_order::Bool=true,
        same_order::Bool=false,
        discount_outliers::Bool=true,
        max_spin_count::Integer=10
    )::Tuple{AbstractVector{<:Integer}, AbstractVector{<:Integer}}

Compute rows and columns orders which move high values close to the diagonal.

For a matrix expressing the cross-similarity between two (possibly different) sets of entities, this produces better
results than clustering. This is because clustering does not care about the order of each two sub-partitions. That is,
clustering is as happy with `((2, 1), (4, 3))` as it is with the more sensible `((1, 2), (3, 4))`. As a result,
visualizations of similarities using naive clustering can be misleading.

This situation is worse in Python or R than it is in Julia, which mercifully provides the `:barjoseph` method to reorder
the branches. Still, the method here may be "more suitable" in specific circumstances (when the data depicts a clear
gradient).

# Parameters

  - `data`: A rectangular matrix containing non-negative values (may be negative if `squared_order`).
  - `order_rows`: Whether to reorder the rows.
  - `order_cols`: Whether to reorder the columns.
  - `squared_order`: Whether to reorder to minimize the l2 norm (otherwise minimizes the l1 norm).
  - `same_order`: Whether to apply the same order to both rows and columns.
  - `discount_outliers`: Whether to do a final order phase discounting outlier values far from the diagonal.
  - `max_spin_count`: How many times to retry improving the solution before giving up.

# Returns

A tuple with two vectors, which contain the order of the rows and the columns.
"""
function slanted_orders(
    data::AbstractMatrix{<:Real};
    order_rows::Bool = true,
    order_cols::Bool = true,
    squared_order::Bool = true,
    same_order::Bool = false,
    discount_outliers::Bool = true,
    max_spin_count::Integer = 10,
)::Tuple{AbstractVector{<:Integer}, AbstractVector{<:Integer}}
    @assert !same_order || (order_rows && order_cols)

    rows_count, cols_count = size(data)

    row_indices = collect(1:rows_count)
    col_indices = collect(1:cols_count)

    best_rows_permutation = copy(row_indices)
    best_cols_permutation = copy(col_indices)

    if same_order
        @assert rows_count == cols_count
        permutation = copy(row_indices)
    end

    if order_rows || order_cols
        if squared_order
            data = data .* data
        end
        @assert minimum(data) >= 0

        function reorder_phase()
            rows_permutation = copy(best_rows_permutation)
            cols_permutation = copy(best_cols_permutation)
            spinning_rows_count = 0
            spinning_cols_count = 0
            spinning_same_count = 0
            was_changed = true
            error_rows = nothing
            error_cols = nothing
            error_same = nothing

            while was_changed
                was_changed = false
                ideal_index = nothing

                if order_rows
                    # For each row, compute its ideal index based on weighted average
                    sum_indexed_rows = vec(sum(data .* transpose(col_indices); dims = 2))
                    sum_squared_rows = vec(sum(data; dims = 2))
                    sum_squared_rows[sum_squared_rows .<= 0] .= 1
                    ideal_row_index = sum_indexed_rows ./ sum_squared_rows

                    if same_order
                        ideal_index = ideal_row_index
                    else
                        ideal_row_index = ideal_row_index .* (rows_count / cols_count)
                        new_rows_permutation = sortperm(ideal_row_index)
                        error = new_rows_permutation .- ideal_row_index
                        new_error_rows = sum(error .* error)
                        new_changed = any(new_rows_permutation .!= row_indices)

                        if isnothing(error_rows) || new_error_rows < error_rows
                            error_rows = new_error_rows
                            spinning_rows_count = 0
                            best_rows_permutation = rows_permutation[new_rows_permutation]
                        else
                            spinning_rows_count += 1
                        end

                        if new_changed && spinning_rows_count < max_spin_count
                            was_changed = true
                            data = data[new_rows_permutation, :]
                            rows_permutation = rows_permutation[new_rows_permutation]
                            row_indices = collect(1:rows_count)  # Reset row indices
                        end
                    end
                end

                if order_cols
                    # For each column, compute its ideal index based on weighted average
                    sum_indexed_cols = vec(sum(data .* row_indices; dims = 1))
                    sum_squared_cols = vec(sum(data; dims = 1))
                    sum_squared_cols[sum_squared_cols .<= 0] .= 1
                    ideal_col_index = sum_indexed_cols ./ sum_squared_cols

                    if same_order
                        ideal_index = (ideal_index + ideal_col_index) ./ 2
                    else
                        ideal_col_index = ideal_col_index .* (cols_count / rows_count)
                        new_cols_permutation = sortperm(ideal_col_index)
                        error = new_cols_permutation .- ideal_col_index
                        new_error_cols = sum(error .* error)
                        new_changed = any(new_cols_permutation .!= col_indices)

                        if isnothing(error_cols) || new_error_cols < error_cols
                            error_cols = new_error_cols
                            spinning_cols_count = 0
                            best_cols_permutation = cols_permutation[new_cols_permutation]
                        else
                            spinning_cols_count += 1
                        end

                        if new_changed && spinning_cols_count < max_spin_count
                            was_changed = true
                            data = data[:, new_cols_permutation]
                            cols_permutation = cols_permutation[new_cols_permutation]
                            col_indices = collect(1:cols_count)  # Reset col indices
                        end
                    end
                end

                if !isnothing(ideal_index)
                    new_permutation = sortperm(ideal_index)
                    error = new_permutation .- ideal_index
                    new_error_same = sum(error .* error)
                    new_changed = any(new_permutation .!= row_indices)

                    if isnothing(error_same) || new_error_same < error_same
                        error_same = new_error_same
                        spinning_same_count = 0
                        best_permutation = permutation[new_permutation]
                        best_rows_permutation = best_permutation
                        best_cols_permutation = best_permutation
                    else
                        spinning_same_count += 1
                    end

                    if new_changed && spinning_same_count < max_spin_count
                        was_changed = true
                        data = data[new_permutation, new_permutation]
                        permutation = permutation[new_permutation]
                        rows_permutation = permutation
                        cols_permutation = permutation
                        row_indices = collect(1:rows_count)  # Reset row indices
                        col_indices = collect(1:cols_count)  # Reset col indices
                    end
                end
            end
        end

        reorder_phase()

        if discount_outliers
            data = data[best_rows_permutation, best_cols_permutation]

            # Create matrices of indices for easy distance calculation
            row_indices_matrix = repeat(row_indices, 1, cols_count)
            col_indices_matrix = transpose(repeat(col_indices, 1, rows_count))

            rows_per_col = rows_count / cols_count
            cols_per_row = cols_count / rows_count

            ideal_row_indices_matrix = col_indices_matrix .* rows_per_col
            ideal_col_indices_matrix = row_indices_matrix .* cols_per_row

            row_distance_matrix = row_indices_matrix .- ideal_row_indices_matrix
            col_distance_matrix = col_indices_matrix .- ideal_col_indices_matrix

            weight_matrix = (1 .+ abs.(row_distance_matrix)) .* (1 .+ abs.(col_distance_matrix))
            data = data ./ weight_matrix

            reorder_phase()
        end
    end

    return (best_rows_permutation, best_cols_permutation)
end

"""
    slanted_reorder(
        data::AbstractMatrix{T};
        order_data::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
        order_rows::Bool=true,
        order_cols::Bool=true,
        squared_order::Bool=true,
        same_order::Bool=false,
        discount_outliers::Bool=true
    )::AbstractMatrix{T} where {T<:Real}

Reorder data rows and columns to move high values close to the diagonal.

Given a matrix expressing the cross-similarity between two (possibly different) sets of entities, this uses
[`slanted_orders`](@ref) to compute the "best" order for visualizing the matrix, then returns the reordered data.

# Parameters

  - `data`: A rectangular matrix to reorder, of non-negative values (unless `order_data` is specified, or `squared_order`)).
  - `order_data`: An optional matrix of non-negative values of the same size to use for computing the orders (may be
    negative if `squared_order`).
  - `order_rows`: Whether to reorder the rows.
  - `order_cols`: Whether to reorder the columns.
  - `squared_order`: Whether to reorder to minimize the l2 norm (otherwise minimizes the l1 norm).
  - `same_order`: Whether to apply the same order to both rows and columns.
  - `discount_outliers`: Whether to do a final order phase discounting outlier values far from the diagonal.

# Returns

A matrix of the same shape whose rows and columns are a permutation of the input.
"""
function slanted_reorder(
    data::AbstractMatrix{T};
    order_data::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    order_rows::Bool = true,
    order_cols::Bool = true,
    squared_order::Bool = true,
    same_order::Bool = false,
    discount_outliers::Bool = true,
)::AbstractMatrix{T} where {T <: Real}
    if isnothing(order_data)
        order_data = data
    end
    @assert size(order_data) == size(data)

    rows_order, columns_order =
        slanted_orders(order_data; order_rows, order_cols, squared_order, same_order, discount_outliers)

    return data[rows_order, columns_order]
end

"""
    reorder_hclust(clusters, order)

Given a clustering of some data, and some ideal order we'd like to use to visualize it, reorder
(but do not modify) the clustering to be as consistent as possible with this ideal order.

# Parameters

  - `clusters`: The existing clustering of the data.
  - `order`: The ideal order we'd like to see the data in.

# Returns

A reordered clustering which is consistent, wherever possible, with the ideal order.
"""
function reorder_hclust(clusters::Hclust{T}, order::AbstractVector{<:Integer})::Hclust{T} where {T <: Real}
    old_of_new = order
    new_of_old = invperm(old_of_new)

    merges = clusters.merges
    merges_count = size(merges, 1)
    merges_data = Vector{Dict}(undef, merges_count)

    for merge_index in 1:merges_count
        a_index = merges[merge_index, 1]
        b_index = merges[merge_index, 2]

        if a_index < 0
            a_indices = [-a_index]
            a_center = new_of_old[-a_index]
        else
            a_data = merges_data[a_index]
            a_indices = a_data[:indices]
            a_center = a_data[:center]
        end

        if b_index < 0
            b_indices = [-b_index]
            b_center = new_of_old[-b_index]
        else
            b_data = merges_data[b_index]
            b_indices = b_data[:indices]
            b_center = b_data[:center]
        end

        a_members = length(a_indices)
        b_members = length(b_indices)

        merged_center = (a_members * a_center + b_members * b_center) / (a_members + b_members)

        if a_center < b_center
            merged_indices = vcat(a_indices, b_indices)
        else
            merges[merge_index, 1] = b_index
            merges[merge_index, 2] = a_index
            merged_indices = vcat(b_indices, a_indices)
        end

        merges_data[merge_index] = Dict(:indices => merged_indices, :center => merged_center)
    end

    new_order = merges_data[merges_count][:indices]

    return Hclust(clusters.merges, clusters.heights, new_order, clusters.linkage)
end

end  # module
