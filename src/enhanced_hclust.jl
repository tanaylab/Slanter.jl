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
import Clustering.hclust_perm
import Clustering.HclustMerges
import Clustering.HclustTrees
import Clustering.MaximumDistance
import Clustering.merge_trees!
import Clustering.MinimalDistance
import Clustering.nearest_neighbor
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
function orderbranches_bypositions!(hmer::HclustMerges, positions::AbstractVector{<:Real})::Nothing
    node_summaries = Vector{Tuple{Float64, Int}}(undef, nnodes(hmer) - 1)  # sum and num of positions of leaves of each node

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

function hclust_ordered(
    d::AbstractMatrix,
    order::AbstractVector{<:Integer},
    metric::ReducibleMetric{T},
)::HclustMerges{T} where {T <: Real}
    dd = copyto!(Matrix{T}(undef, size(d)...), d)
    htre = HclustTrees{T}(size(d, 1))
    tree_in_position = copyto!(Vector{Int}(undef, length(order)), order)
    position_of_tree = invperm(tree_in_position)
    return ordered_clustering(dd, htre, metric, tree_in_position, position_of_tree)
end

function ordered_clustering(
    dd::AbstractMatrix,
    htre::HclustTrees{T},
    metric::ReducibleMetric{T},
    tree_in_position::Vector{Int},
    position_of_tree::Vector{Int},
)::HclustMerges{T} where {T <: Real}
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

        ## Needed because, for example, minimal distance can actually go down when we merge more trees (if grouped).
        min_height = 0
        for tree_id in (htre.id[NN_l_tree], htre.id[NN_r_tree])
            if tree_id > 0
                min_height = max(min_height, htre.merges.heights[tree_id])
            end
        end

        last_tree = ntrees(htre)
        ## update the distance matrix (while the trees are not merged yet)
        ## TODO: This isn't optimal. We only need the distances of adjacent trees.
        update_distances_upon_merge!(dd, metric, i -> tree_size(htre, i), NN_lo_tree, NN_hi_tree, last_tree)
        merge_trees!(htre, NN_l_tree, NN_r_tree, NNmindist) # side effect: puts last_tree to NN_r_tree

        htre.merges.heights[end] = max(htre.merges.heights[end], min_height)

        position_of_tree[NN_r_tree] = position_of_tree[last_tree]
        position_of_tree[position_of_tree .> NN_r_pos] .-= 1
        pop!(position_of_tree)

        ## TODO: This probably isn't optimal either (at minimum, we should be able to avoid the memory allocation).
        tree_in_position = invperm(position_of_tree)
    end
    return htre.merges
end

function hclust_basic(d::AbstractMatrix, metric::ReducibleMetric{T})::HclustMerges{T} where {T <: Real}
    dd = copyto!(Matrix{T}(undef, size(d)...), d)
    htre = HclustTrees{T}(size(d, 1))
    return complete_clustering(dd, htre, metric)
end

function complete_clustering(
    dd::AbstractMatrix,
    htre::HclustTrees{T},
    metric::ReducibleMetric{T},
)::HclustMerges{T} where {T <: Real}
    NN = [1]      # nearest neighbors chain of tree indices, init by random tree index
    while ntrees(htre) > 1
        isempty(NN) && push!(NN, 1) # restart NN chain
        # search for a pair of closest clusters,
        # they would be mutual nearest neighbors on top of the NN stack
        NNmindist = typemax(T)
        while true
            ## find NNnext: nearest neighbor of NN[end] (and the next stack top)
            NNnext, NNmindist = nearest_neighbor(dd, NN[end], ntrees(htre))
            @assert NNnext > 0
            if length(NN) > 1 && NNnext == NN[end - 1] # NNnext==NN[end-1] and NN[end] are mutual n.neighbors
                break
            else
                push!(NN, NNnext)
            end
        end
        ## merge NN[end] and its nearest neighbor, i.e., NN[end-1]
        NNlo = pop!(NN)
        NNhi = pop!(NN)
        if NNlo > NNhi
            NNlo, NNhi = NNhi, NNlo
        end

        ## Needed because, for example, minimal distance can actually go down when we merge more trees (if grouped).
        min_height = 0
        for tree_id in (htre.id[NNlo], htre.id[NNhi])
            if tree_id > 0
                min_height = max(min_height, htre.merges.heights[tree_id])
            end
        end

        last_tree = ntrees(htre)
        ## update the distance matrix (while the trees are not merged yet)
        update_distances_upon_merge!(dd, metric, i -> tree_size(htre, i), NNlo, NNhi, last_tree)
        merge_trees!(htre, NNlo, NNhi, NNmindist) # side effect: puts last_tree to NNhi

        htre.merges.heights[end] = max(htre.merges.heights[end], min_height)

        for k in eachindex(NN)
            NNk = NN[k]
            if (NNk == NNlo) || (NNk == NNhi)
                # in case of duplicate distances, NNlo or NNhi may appear in NN
                # several times, if that's detected, restart NN search
                empty!(NN)  # UNTESTED
                break  # UNTESTED
            elseif NNk == last_tree
                ## the last_tree was moved to NNhi slot by merge_trees!()
                # update the NN references to it
                NN[k] = NNhi  # UNTESTED
            end
        end
        isempty(NN) && push!(NN, 1) # restart NN chain
    end
    return htre.merges
end

function hclust_grouped(
    d::AbstractMatrix,
    groups::AbstractVector{<:AbstractString},
    metric::ReducibleMetric{T},
)::HclustMerges{T} where {T <: Real}
    dd = copyto!(Matrix{T}(undef, size(d)...), d)
    htre = HclustTrees{T}(size(d, 1))
    unique_groups = unique(groups)
    group_of_tree = zeros(Int, ntrees(htre))
    for (group_index, group) in enumerate(unique_groups)
        group_of_tree[groups .== group] .= group_index
    end
    for group_index in 1:length(unique_groups)
        grouped_clustering(dd, htre, metric, group_of_tree, group_index)  # NOJET
    end
    return complete_clustering(dd, htre, metric)
end

function hclust_grouped(
    d::AbstractMatrix,
    groups::AbstractVector{<:Integer},
    metric::ReducibleMetric{T},
)::HclustMerges{T} where {T <: Real}
    dd = copyto!(Matrix{T}(undef, size(d)...), d)
    htre = HclustTrees{T}(size(d, 1))
    n_groups = maximum(groups)
    group_of_tree = copyto!(Vector{Int}(undef, length(groups)), groups)
    for group_index in 1:n_groups
        grouped_clustering(dd, htre, metric, group_of_tree, group_index)  # NOJET
    end

    position_of_tree = group_of_tree
    tree_in_position = invperm(position_of_tree)

    return ordered_clustering(dd, htre, metric, tree_in_position, position_of_tree)
end

function group_nearest_neighbor(d::AbstractMatrix, i::Integer, N::Integer, group_of_tree::Vector{Int})
    (N <= 1) && return 0, NaN

    NNi = 0
    NNdist = 0
    group = group_of_tree[i]

    @inbounds for j in 1:length(group_of_tree)
        @inbounds if j != i && group_of_tree[j] == group
            @inbounds dist = d[min(j, i), max(j, i)]
            if NNi == 0 || NNdist > dist
                NNi = j
                NNdist = dist
            end
        end
    end

    @assert NNi != 0
    return NNi, NNdist
end

function grouped_clustering(
    dd::AbstractMatrix,
    htre::HclustTrees{T},
    metric::ReducibleMetric{T},
    group_of_tree::Vector{Int},
    group_index::Int,
)::Nothing where {T <: Real}
    NN = Int[]      # nearest neighbors chain of tree indices, init by random tree index
    n_group_trees = sum(group_of_tree .== group_index)
    while n_group_trees > 1
        isempty(NN) && push!(NN, findfirst(group_of_tree .== group_index)) # restart NN chain
        # search for a pair of closest clusters,
        # they would be mutual nearest neighbors on top of the NN stack
        NNmindist = typemax(T)
        while true
            ## find NNnext: nearest neighbor of NN[end] (and the next stack top)
            NNnext, NNmindist = group_nearest_neighbor(dd, NN[end], ntrees(htre), group_of_tree)
            @assert NNnext > 0
            if length(NN) > 1 && NNnext == NN[end - 1] # NNnext==NN[end-1] and NN[end] are mutual n.neighbors
                break
            else
                push!(NN, NNnext)
            end
        end
        ## merge NN[end] and its nearest neighbor, i.e., NN[end-1]
        NNlo = pop!(NN)
        NNhi = pop!(NN)
        if NNlo > NNhi
            NNlo, NNhi = NNhi, NNlo
        end
        last_tree = ntrees(htre)
        ## update the distance matrix (while the trees are not merged yet)
        update_distances_upon_merge!(dd, metric, i -> tree_size(htre, i), NNlo, NNhi, last_tree)
        merge_trees!(htre, NNlo, NNhi, NNmindist) # side effect: puts last_tree to NNhi
        group_of_tree[NNhi] = group_of_tree[last_tree]
        pop!(group_of_tree)
        n_group_trees -= 1
        for k in eachindex(NN)
            NNk = NN[k]
            if (NNk == NNlo) || (NNk == NNhi)
                # in case of duplicate distances, NNlo or NNhi may appear in NN
                # several times, if that's detected, restart NN search
                empty!(NN)  # UNTESTED
                break  # UNTESTED
            elseif NNk == last_tree
                ## the last_tree was moved to NNhi slot by merge_trees!()
                # update the NN references to it
                NN[k] = NNhi
            end
        end
    end
    return nothing
end

"""
    ehclust(d::AbstractMatrix; [linkage], [uplo], [branchorder], [order]) -> Hclust

Enhanced [hclust]*https://github.com/JuliaStats/Clustering.jl/blob/master/src/hclust.jl).
This is similar to `hclust` with the following extensions:

  - If `branchorder` is a vector of `Real` numbers, one per leaf, then we reorder the branches so that each leaf position
    would be as close as possible to its `branchorder` value. Technically we compute a center of gravity for each node and
    reorder the tree such that that at each branch, the left sub-tree center of gravity is to the left (lower than) the
    center of gravity of the right sub-tree.
  - If `order` is specified, it must be a permutation of the 1:N leaf indices. This will be the final order of the result;
    that is, we constrain the tree so that each node covers a continuous range of leaves (by this order). If you specify
    an explicit `branchorder`, this will rotate some nodes so the result will no longer be in the specified order, but
    the tree is still constrained as above.
  - If `groups` is a vector of strings, then we first cluster all the entries for each group together, then combine the
    results. This is mutually exclusive with specifying an `order`.
  - If `groups` is a vector of integers, they are expected to cover a range 1:N. We again cluster each group separately,
    and then cluster the groups enforcing them to be in ascending order. Applying `branchorder` in this case will only
    reorder branches inside each group, preserving the ascending order between the groups.

The `groups` and `order` parameters are mutually exclusive.

```jldoctest
using Test
using Clustering
using Distances

data = rand(4, 10)
distances = pairwise(Euclidean(), data; dims = 2)
result = hclust(distances)
eresult = ehclust(distances)
@test result.order == eresult.order

result = hclust(distances; linkage = :ward)
eresult = ehclust(distances; linkage = :ward)
@test result.order == eresult.order

println("OK")

# output

OK
```

```jldoctest
using Test
using Distances

data = rand(4, 10)
distances = pairwise(Euclidean(), data; dims = 2)
positions = rand(10)
result = ehclust(distances; branchorder = positions)
merges_data = Vector{Tuple{Int, Float64}}(undef, 9)
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
result = ehclust(distances; branchorder = :r, order)
@test result.order != order

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
groups = ["A", "B"][rand(1:2, 10)]
result = ehclust(distances; groups)
a_indices = findall(groups[result.order] .== "A")
b_indices = findall(groups[result.order] .== "B")
@assert maximum(a_indices) < minimum(b_indices) || maximum(b_indices) < minimum(a_indices)

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

groups = rand(1:2, 10)
result = ehclust(distances; groups)
one_indices = findall(groups[result.order] .== 1)
two_indices = findall(groups[result.order] .== 2)
@assert maximum(one_indices) < minimum(two_indices)

groups = 3 .- groups
result = ehclust(distances; groups)
one_indices = findall(groups[result.order] .== 1)
two_indices = findall(groups[result.order] .== 2)
@assert maximum(one_indices) < minimum(two_indices)

bresult = ehclust(distances; branchorder = :r, groups)
@assert bresult.order != result.order
one_indices = findall(groups[bresult.order] .== 1)
two_indices = findall(groups[bresult.order] .== 2)
@assert maximum(one_indices) < minimum(two_indices)

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
    groups::Union{AbstractVector{<:AbstractString}, AbstractVector{<:Integer}, Nothing} = nothing,
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

    if groups !== nothing
        @assert length(groups) == size(d, 1)
    end

    @assert order === nothing || groups === nothing

    if linkage == :single
        if order !== nothing
            hmer = hclust_ordered(sd, order, MinimalDistance(sd))
        elseif groups !== nothing
            hmer = hclust_grouped(sd, groups, MinimalDistance(sd))
        else
            hmer = hclust_minimum(sd)
        end

    elseif linkage == :complete
        if order !== nothing  # UNTESTED
            hmer = hclust_ordered(sd, order, MaximumDistance(sd))  # UNTESTED
        elseif groups !== nothing  # UNTESTED
            hmer = hclust_grouped(sd, groups, MaximumDistance(sd))  # UNTESTED
        else
            hmer = hclust_basic(sd, MaximumDistance(sd))  # UNTESTED
        end

    elseif linkage == :average
        if order !== nothing  # UNTESTED
            hmer = hclust_ordered(sd, order, AverageDistance(sd))  # UNTESTED
        elseif groups !== nothing  # UNTESTED
            hmer = hclust_grouped(sd, groups, AverageDistance(sd))  # UNTESTED
        else
            hmer = hclust_basic(sd, AverageDistance(sd))  # UNTESTED
        end

    elseif linkage == :ward_presquared
        if order !== nothing  # UNTESTED
            hmer = hclust_ordered(sd, order, WardDistance(sd))  # UNTESTED
        elseif groups !== nothing  # UNTESTED
            hmer = hclust_grouped(sd, groups, WardDistance(sd))  # UNTESTED
        else
            hmer = hclust_basic(sd, WardDistance(sd))  # UNTESTED
        end

    elseif linkage == :ward
        if sd === d
            sd = abs2.(sd)
        else
            sd .= abs2.(sd)  # UNTESTED
        end
        if order !== nothing
            hmer = hclust_ordered(sd, order, WardDistance(sd))  # UNTESTED
        elseif groups !== nothing
            hmer = hclust_grouped(sd, groups, WardDistance(sd))  # UNTESTED
        else
            hmer = hclust_basic(sd, WardDistance(sd))
        end
        hmer.heights .= sqrt.(hmer.heights)

    else
        throw(ArgumentError("Unsupported cluster linkage $linkage"))  # UNTESTED
    end

    if branchorder === nothing && order === nothing && !(groups isa AbstractVector{<:Integer})
        branchorder = :r
    end

    if branchorder == :barjoseph || branchorder == :optimal
        orderbranches_barjoseph!(hmer, sd)  # NOJET  # UNTESTED
    elseif branchorder == :r
        orderbranches_r!(hmer)
    elseif branchorder isa AbstractVector{<:Real}
        @assert length(branchorder) == size(d, 1)
        orderbranches_bypositions!(hmer, branchorder)
    elseif branchorder !== nothing
        throw(ArgumentError("Unsupported branchorder=$branchorder method"))  # UNTESTED
    end

    if branchorder !== nothing && groups isa AbstractVector{<:Integer}
        branchorder = sortperm(collect(zip(groups, hclust_perm(hmer))))
        orderbranches_bypositions!(hmer, invperm(branchorder))
    end

    return Hclust(hmer, linkage)
end

end
