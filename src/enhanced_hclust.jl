"""
Enhancements of [hclust]*https://github.com/JuliaStats/Clustering.jl/blob/master/src/hclust.jl)
"""
module EnhancedHclust

export ehclust

using Clustering
using LinearAlgebra

import Clustering.assertdistancematrix
import Clustering.AverageDistance
import Clustering.hclust_minimum
import Clustering.hclust_nn_lw
import Clustering.HclustMerges
import Clustering.HclustTrees
import Clustering.MaximumDistance
import Clustering.merge_trees!
import Clustering.MinimalDistance
import Clustering.nnodes
import Clustering.ntrees
import Clustering.orderbranches_barjoseph!
import Clustering.orderbranches_r!
import Clustering.ReducibleMetric
import Clustering.Symmetric
import Clustering.tree_size
import Clustering.update_distances_upon_merge!
import Clustering.WardDistance

## Given a hierarchical cluster and some ideal positions of the leaves,
## reorder the branches so they will be the most compatible with these positions.
##
## This modifies the `Hclust` in-place, instead of the normal API of  `orderbranches_...`,
## because we
function orderbranches_byorder!(hmer::HclustMerges, positions::AbstractVector{<:Real})::Nothing
    node_summaries = Vector{Tuple{Float32, Int32}}(undef, nnodes(hmer) - 1)  # sum and num of positions of leaves of each node

    for v in 1:(nnodes(hmer) - 1)
        @inbounds vl = hmer.mleft[v]
        @inbounds vr = hmer.mright[v]

        if vl < 0
            @inbounds l_center = l_sum = positions[-vl]
            l_num = 1
        else
            @inbounds l_sum, l_num = node_summaries[vl]
            l_center = l_sum / l_num
        end

        if vr < 0
            @inbounds r_center = r_sum = positions[-vr]
            r_num = 1
        else
            @inbounds r_sum, r_num = node_summaries[vr]
            r_center = r_sum / r_num
        end

        if l_center > r_center
            @inbounds hmer.mleft[v] = vr
            @inbounds hmer.mright[v] = vl
        end

        @inbounds node_summaries[v] = (l_sum + r_sum, l_num + r_num)
    end

    return nothing
end

function oclust(d::AbstractMatrix, order::AbstractVector{<:Integer}, metric::ReducibleMetric{T}) where {T <: Real}
    dd = copyto!(Matrix{T}(undef, size(d)...), d)
    htre = HclustTrees{T}(size(d, 1))
    tree_in_position = copyto!(Vector{Int32}(undef, length(order)), order)
    position_of_trees = invperm(tree_in_position)
    max_height = 0
    while ntrees(htre) > 1
        # Find the nearest adjacent trees in-order.
        NN_r_pos = 0
        NN_l_tree = 0
        NN_r_tree = 0
        NN_lo_tree = 0
        NN_hi_tree = 0
        NNmindist = typemax(T)

        for i in 1:(ntrees(htre) - 1)
            l_pos = i
            r_pos = i + 1
            l_tree = tree_in_position[l_pos]
            r_tree = tree_in_position[r_pos]
            if l_tree < r_tree
                lo_tree, hi_tree = l_tree, r_tree
            else
                lo_tree, hi_tree = r_tree, l_tree
            end
            dist = dd[lo_tree, hi_tree]

            if l_pos == 1 || dist < NNmindist
                NNmindist = dist
                NN_r_pos = r_pos
                NN_l_tree = l_tree
                NN_r_tree = r_tree
                NN_lo_tree = lo_tree
                NN_hi_tree = hi_tree
            end
        end
        last_tree = ntrees(htre)
        ## update the distance matrix (while the trees are not merged yet)
        ## TODO: This isn't optimal. We only need the distances of adjacent trees.
        update_distances_upon_merge!(dd, metric, i -> tree_size(htre, i), NN_lo_tree, NN_hi_tree, last_tree)
        merge_trees!(htre, NN_l_tree, NN_r_tree, NNmindist) # side effect: puts last_tree to NN_r_tree

        max_height = htre.merges.heights[end] = max(max_height, htre.merges.heights[end])  ## Force non-decreasing heights

        position_of_trees[NN_r_tree] = position_of_trees[last_tree]
        position_of_trees[position_of_trees .> NN_r_pos] .-= 1
        pop!(position_of_trees)

        ## TODO: This probably isn't optimal either (at minimum, we should be able to avoid the memory allocation.
        tree_in_position = invperm(position_of_trees)
    end
    return htre.merges
end

"""
    ehclust(d::AbstractMatrix; [linkage], [uplo], [branchorder], [order]) -> Hclust

Enhanced [hclust]*https://github.com/JuliaStats/Clustering.jl/blob/master/src/hclust.jl).
This is similar to `hclust` with the following extensions:

  - If `branchorder` is a vector of `Real` numbers, one per leaf, then we reorder the branches so that each leaf position
    would be as close as possible to its `branchorder` value. Technically we compute a center of gravity for each node and
    reorder the tree such that that at each branch, the left sub-tree center of gravity is to the left (lower than) the
    center of gravity of the right sub-tree.

  - If `order` is specified, it must be a permutation of the 1:N leaf indices. This will be the final order of the result.
    If you specify an explicit `branchorder`, this will modify the result.

```jldoctest
using Test
using Distances

data = rand(4, 10)
distances = pairwise(Euclidean(), data; dims = 2)
positions = rand(10)
result = ehclust(distances; branchorder = positions)
merges_data = Vector{Tuple{Int32, Float32}}(undef, 9)
for merge_index in 1:9
    left = result.merges[merge_index, 1]
    if left < 0
        left_size = 1
        left_center = positions[-left]
    else
        left_size, left_center = merges_data[left]
    end

    right = result.merges[merge_index, 2]
    if right < 0
        right_size = 1
        right_center = positions[-right]
    else
        right_size, right_center = merges_data[right]
    end

    @test left_center <= right_center
    merged_size = left_size + right_size
    merged_center = (left_center * left_size + right_center * right_size) / merged_size
    merges_data[merge_index] = (merged_size, merged_center)
end

println("OK")

# output

OK
```

```jldoctest
using Test
using Distances
using Random

data = rand(4, 10)
distances = pairwise(Euclidean(), data; dims = 2)
order = collect(1:10)
shuffle!(order)
result = ehclust(distances; order)
@test result.order == order
result = ehclust(distances; branchorder = :r)
result = ehclust(distances; branchorder = :r, order)
@test result.order != order

println("OK")

# output

OK
```
"""
function ehclust(
    d::AbstractMatrix;
    linkage::Symbol = :single,
    uplo::Union{Symbol, Nothing} = nothing,
    branchorder::Union{Symbol, AbstractVector{<:Real}, Nothing} = nothing,
    order::Union{AbstractVector{<:Integer}, Nothing} = nothing,
)::Hclust
    if uplo !== nothing
        sd = Symmetric(d, uplo) # use upper/lower part of d  # NOJET # UNTESTED
    else
        assertdistancematrix(d)
        sd = d
    end

    if order !== nothing
        @assert length(order) == size(d, 1)
    end

    if linkage == :single
        if order === nothing
            hmer = hclust_minimum(sd)
        else
            hmer = oclust(sd, order, MinimalDistance(sd))
        end

    elseif linkage == :complete  # UNTESTED
        if order === nothing
            hmer = hclust_nn_lw(sd, MaximumDistance(sd))  # UNTESTED
        else
            hmer = oclust(sd, order, MaximumDistance(sd))
        end

    elseif linkage == :average  # UNTESTED
        if order === nothing
            hmer = hclust_nn_lw(sd, AverageDistance(sd))  # UNTESTED
        else
            hmer = oclust(sd, order, MaximumDistance(sd))
        end

    elseif linkage == :ward_presquared  # UNTESTED
        if order === nothing
            hmer = hclust_nn_lw(sd, WardDistance(sd))  # UNTESTED
        else
            hmer = oclust(sd, order, WardDistance(sd))
        end

    elseif linkage == :ward  # UNTESTED
        if sd === d  # UNTESTED
            sd = abs2.(sd)  # UNTESTED
        else
            sd .= abs2.(sd)  # UNTESTED
        end
        if order === nothing
            hmer = hclust_nn_lw(sd, WardDistance(sd))  # UNTESTED
        else
            hmer = oclust(sd, order, WardDistance(sd))
        end
        hmer.heights .= sqrt.(hmer.heights)  # UNTESTED

    else
        throw(ArgumentError("Unsupported cluster linkage $linkage"))  # UNTESTED
    end

    if branchorder === nothing && order === nothing
        branchorder = :r
    end

    if branchorder == :barjoseph || branchorder == :optimal
        orderbranches_barjoseph!(hmer, sd)  # NOJET  # UNTESTED
    elseif branchorder == :r
        orderbranches_r!(hmer)  # UNTESTED
    elseif branchorder isa AbstractVector{<:Real}
        @assert length(branchorder) == size(d, 1)
        orderbranches_byorder!(hmer, branchorder)
    elseif branchorder !== nothing
        throw(ArgumentError("Unsupported branchorder=$branchorder method"))  # UNTESTED
    end

    return Hclust(hmer, linkage)
end

end
