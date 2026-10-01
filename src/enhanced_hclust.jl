"""
Enhancements of [hclust]*https://github.com/JuliaStats/Clustering.jl/blob/master/src/hclust.jl)
"""
module EnhancedHclust

export ehclust

using Clustering
using LinearAlgebra

import Clustering.assertdistancematrix
import Clustering.AverageDistance  # NOLINT
import Clustering.hclust_minimum
import Clustering.hclust_perm
import Clustering.HclustMerges  # NOLINT
import Clustering.HclustTrees  # NOLINT
import Clustering.MaximumDistance  # NOLINT
import Clustering.merge_trees!
import Clustering.MinimalDistance  # NOLINT
import Clustering.nearest_neighbor
import Clustering.nnodes
import Clustering.ntrees
import Clustering.orderbranches_barjoseph!
import Clustering.orderbranches_r!
import Clustering.ReducibleMetric  # NOLINT
import Clustering.Symmetric
import Clustering.tree_size
import Clustering.update_distances_upon_merge!
import Clustering.WardDistance  # NOLINT

## Given a hierarchical cluster and some ideal positions of the leaves,
## reorder the branches so they will be the most compatible with these positions.
##
## This modifies the `Hclust` in-place, instead of the normal API of  `orderbranches_...`,
## because we
function orderbranches_bypositions!(hmer::HclustMerges, positions::AbstractVector{<:Real})::Nothing
    # Sum and num of positions of leaves of each node.
    node_summaries = Vector{Tuple{Float64, Int}}(undef, nnodes(hmer) - 1)

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

## The key to sort each leaf by, to restore the order of a level after the branches were ordered. A numbered level is
## sorted by its numbers; a named level, or a missing one, is sorted by the position the clustering gave the first leaf
## of each of its entries, which leaves it where it was placed.
function level_sort_key(::Nothing, positions::AbstractVector{<:Integer})::AbstractVector{<:Integer}
    return zeros(Int, length(positions))
end

function level_sort_key(level::AbstractVector{<:Integer}, ::AbstractVector{<:Integer})::AbstractVector{<:Integer}
    return level
end

function level_sort_key(
    level::AbstractVector{<:AbstractString},
    positions::AbstractVector{<:Integer},
)::AbstractVector{<:Integer}
    first_position_per_value = Dict{AbstractString, Int}()
    for (value, position) in zip(level, positions)
        first_position_per_value[value] = min(get(first_position_per_value, value, position), position)
    end
    return [first_position_per_value[value] for value in level]
end

## Cluster by one level of groups, or by two when the entries are also assigned to subgroups nested in these groups.
function grouped_clustering_of(
    d::AbstractMatrix,
    groups::AbstractVector,
    ::Nothing,
    metric::ReducibleMetric{T},
)::HclustMerges{T} where {T <: Real}
    return hclust_grouped(d, groups, metric)
end

function grouped_clustering_of(
    d::AbstractMatrix,
    groups::AbstractVector,
    subgroups::AbstractVector,
    metric::ReducibleMetric{T},
)::HclustMerges{T} where {T <: Real}
    return hclust_grouped(d, groups, subgroups, metric)
end

## The dense 1-based index of the level of each entry. A level specified by names is laid out by the clustering, so any
## consistent indexing of it will do; a level specified by numbers is laid out in the order of these numbers.
function level_index_per_entry(level::AbstractVector{<:AbstractString})::Vector{Int}
    index_per_value = Dict(value => index for (index, value) in enumerate(unique(level)))
    return [index_per_value[value] for value in level]
end

function level_index_per_entry(level::AbstractVector{<:Integer})::Vector{Int}
    return copyto!(Vector{Int}(undef, length(level)), level)
end

## Merge the trees of one group in the order of their positions; that is, `ordered_clustering` restricted to the trees
## of a single group. Both the group and the position of each tree are maintained as the trees are merged, the same way
## `grouped_clustering` maintains the group of each tree.
function ordered_group_clustering(
    dd::AbstractMatrix,
    htre::HclustTrees{T},
    metric::ReducibleMetric{T},
    group_of_tree::Vector{Int},
    position_of_tree::AbstractVector,
    group_index::Int,
)::Nothing where {T <: Real}
    while true
        trees_of_group = findall(==(group_index), group_of_tree)
        length(trees_of_group) <= 1 && break
        sort!(trees_of_group; by = tree -> position_of_tree[tree])

        NNlo = NNhi = 0
        NNmindist = typemax(T)
        for position in 1:(length(trees_of_group) - 1)
            lo_tree, hi_tree = minmax(trees_of_group[position], trees_of_group[position + 1])
            dist = dd[lo_tree, hi_tree]
            if NNlo == 0 || dist < NNmindist
                NNmindist = dist
                NNlo, NNhi = lo_tree, hi_tree
            end
        end

        ## Needed because, for example, minimal distance can actually go down when we merge more trees (if grouped).
        min_height = 0
        for tree_id in (htre.id[NNlo], htre.id[NNhi])
            if tree_id > 0
                min_height = max(min_height, htre.merges.heights[tree_id])
            end
        end

        last_tree = ntrees(htre)
        update_distances_upon_merge!(dd, metric, i -> tree_size(htre, i), NNlo, NNhi, last_tree)
        merge_trees!(htre, NNlo, NNhi, NNmindist)  # side effect: puts last_tree to NNhi
        htre.merges.heights[end] = max(htre.merges.heights[end], min_height)

        ## The merged tree is left in `NNlo` and takes the earlier of the two positions, and `NNhi` is overwritten by
        ## the last tree, which `merge_trees!` moved into it.
        position_of_tree[NNlo] = min(position_of_tree[NNlo], position_of_tree[NNhi])
        group_of_tree[NNhi] = group_of_tree[last_tree]
        position_of_tree[NNhi] = position_of_tree[last_tree]
        pop!(group_of_tree)
        pop!(position_of_tree)
    end
    return nothing
end

## Place the subgroups of each group; named subgroups are placed by the clustering, numbered ones in the order of their
## numbers.
function place_subgroups(
    dd::AbstractMatrix,
    htre::HclustTrees{T},
    metric::ReducibleMetric{T},
    ::AbstractVector{<:AbstractString},
    group_index_of_tree::Vector{Int},
    ::AbstractVector,
    n_groups::Integer,
)::Nothing where {T <: Real}
    for group_index in 1:n_groups
        grouped_clustering(dd, htre, metric, group_index_of_tree, group_index)  # NOJET
    end
    return nothing
end

function place_subgroups(
    dd::AbstractMatrix,
    htre::HclustTrees{T},
    metric::ReducibleMetric{T},
    ::AbstractVector{<:Integer},
    group_index_of_tree::Vector{Int},
    subgroup_of_tree::AbstractVector,
    n_groups::Integer,
)::Nothing where {T <: Real}
    for group_index in 1:n_groups
        ordered_group_clustering(dd, htre, metric, group_index_of_tree, subgroup_of_tree, group_index)
    end
    return nothing
end

## Cluster within each subgroup, then place the subgroups of each group, leaving one tree per group. The `htre` trees
## and the returned group of each of them are left for the caller to place the groups themselves.
function collapse_subgroups(
    dd::AbstractMatrix,
    htre::HclustTrees{T},
    metric::ReducibleMetric{T},
    groups::AbstractVector,
    subgroups::AbstractVector,
)::Vector{Int} where {T <: Real}
    group_index_per_entry = level_index_per_entry(groups)
    n_groups = maximum(group_index_per_entry)

    # A subgroup is nested in its group, so the same subgroup of two different groups is two different subgroups; the
    # identity of a subgroup is therefore its (group, subgroup) pair. This means the subgroups need not be unique - a
    # group may well number its own subgroups 1, 2, 3 just like the next group does.
    pair_per_entry = collect(zip(group_index_per_entry, subgroups))
    index_per_pair = Dict(pair => index for (index, pair) in enumerate(unique(pair_per_entry)))
    subgroup_index_per_entry = [index_per_pair[pair] for pair in pair_per_entry]
    n_subgroups = length(index_per_pair)

    group_index_per_subgroup = Vector{Int}(undef, n_subgroups)
    subgroup_per_subgroup = Vector{eltype(subgroups)}(undef, n_subgroups)
    for (pair, index) in index_per_pair
        group_index_per_subgroup[index] = pair[1]
        subgroup_per_subgroup[index] = pair[2]
    end

    # Collapse each subgroup into a single tree. This maintains the subgroup of each tree, so when it is done there is
    # exactly one tree per subgroup.
    subgroup_index_of_tree = subgroup_index_per_entry
    for subgroup_index in 1:n_subgroups
        grouped_clustering(dd, htre, metric, subgroup_index_of_tree, subgroup_index)  # NOJET
    end

    # Collapse the trees of the subgroups of each group into a single tree, the same way. Numbered subgroups are placed
    # by their own numbers, which are only compared within a group.
    group_index_of_tree = [group_index_per_subgroup[subgroup_index] for subgroup_index in subgroup_index_of_tree]
    subgroup_of_tree = [subgroup_per_subgroup[subgroup_index] for subgroup_index in subgroup_index_of_tree]
    place_subgroups(dd, htre, metric, subgroups, group_index_of_tree, subgroup_of_tree, n_groups)

    return group_index_of_tree
end

## Named groups are placed by the clustering, and numbered groups in the order of their numbers, regardless of how the
## subgroups nested in them are placed. Numbering both levels therefore lays the entries out in the order of their
## (group, subgroup) pair.
function hclust_grouped(
    d::AbstractMatrix,
    groups::AbstractVector{<:AbstractString},
    subgroups::AbstractVector,
    metric::ReducibleMetric{T},
)::HclustMerges{T} where {T <: Real}
    dd = copyto!(Matrix{T}(undef, size(d)...), d)
    htre = HclustTrees{T}(size(d, 1))
    collapse_subgroups(dd, htre, metric, groups, subgroups)
    return complete_clustering(dd, htre, metric)
end

function hclust_grouped(
    d::AbstractMatrix,
    groups::AbstractVector{<:Integer},
    subgroups::AbstractVector,
    metric::ReducibleMetric{T},
)::HclustMerges{T} where {T <: Real}
    dd = copyto!(Matrix{T}(undef, size(d)...), d)
    htre = HclustTrees{T}(size(d, 1))
    position_of_tree = collapse_subgroups(dd, htre, metric, groups, subgroups)
    return ordered_clustering(dd, htre, metric, invperm(position_of_tree), position_of_tree)
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

function group_nearest_neighbor(d::AbstractMatrix, i::Integer, N::Integer, group_of_tree::Vector{Int})  # UNTESTED
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
        ## Needed because, for example, minimal distance can actually go down when we merge more trees (if grouped).
        ## Without this the merges would not be sorted by height, and `orderbranches_r!` would place a merge before
        ## the merges it is made of.
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

  - If `branchorder` is a vector of `Real` numbers, one per leaf, then we reorder the branches so that each leaf
    position would be as close as possible to its `branchorder` value. Technically we compute a center of gravity for
    each node and reorder the tree such that that at each branch, the left sub-tree center of gravity is to the left
    (lower than) the center of gravity of the right sub-tree.
  - If `order` is specified, it must be a permutation of the 1:N leaf indices. This will be the final order of the
    result; that is, we constrain the tree so that each node covers a continuous range of leaves (by this order). If you
    specify an explicit `branchorder`, this will rotate some nodes so the result will no longer be in the specified
    order, but the tree is still constrained as above.
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
    subgroups::Union{AbstractVector{<:AbstractString}, AbstractVector{<:Integer}, Nothing} = nothing,
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

    if subgroups !== nothing
        @assert length(subgroups) == size(d, 1)
        @assert groups !== nothing "specified subgroups without groups"
    end

    @assert order === nothing || groups === nothing

    if linkage == :single
        if order !== nothing
            hmer = hclust_ordered(sd, order, MinimalDistance(sd))
        elseif groups !== nothing
            hmer = grouped_clustering_of(sd, groups, subgroups, MinimalDistance(sd))
        else
            hmer = hclust_minimum(sd)
        end

    elseif linkage == :complete
        if order !== nothing  # UNTESTED
            hmer = hclust_ordered(sd, order, MaximumDistance(sd))  # UNTESTED
        elseif groups !== nothing  # UNTESTED
            hmer = grouped_clustering_of(sd, groups, subgroups, MaximumDistance(sd))  # UNTESTED
        else
            hmer = hclust_basic(sd, MaximumDistance(sd))  # UNTESTED
        end

    elseif linkage == :average
        if order !== nothing  # UNTESTED
            hmer = hclust_ordered(sd, order, AverageDistance(sd))  # UNTESTED
        elseif groups !== nothing  # UNTESTED
            hmer = grouped_clustering_of(sd, groups, subgroups, AverageDistance(sd))  # UNTESTED
        else
            hmer = hclust_basic(sd, AverageDistance(sd))  # UNTESTED
        end

    elseif linkage == :ward_presquared
        if order !== nothing  # UNTESTED
            hmer = hclust_ordered(sd, order, WardDistance(sd))  # UNTESTED
        elseif groups !== nothing  # UNTESTED
            hmer = grouped_clustering_of(sd, groups, subgroups, WardDistance(sd))  # UNTESTED
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
            hmer = grouped_clustering_of(sd, groups, subgroups, WardDistance(sd))  # UNTESTED
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

    ## Ordering the branches may flip a whole sub-tree, which undoes the order of a numbered level; restore it by
    ## sorting the leaves by their level(s) and their current position. A named level is left wherever the clustering
    ## placed it, which is the position of the first of its leaves.
    if groups isa AbstractVector{<:Integer} || subgroups isa AbstractVector{<:Integer}
        positions = hclust_perm(hmer)
        groups_key = level_sort_key(groups, positions)
        subgroups_key = level_sort_key(subgroups, positions)
        branchorder = sortperm(collect(zip(groups_key, subgroups_key, positions)))
        orderbranches_bypositions!(hmer, invperm(branchorder))
    end

    return Hclust(hmer, linkage)
end

end
