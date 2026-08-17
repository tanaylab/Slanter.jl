using Clustering
using Distances
using Documenter
using Random
using Slanter
using Test

Random.seed!(123456)

@testset "doctests" begin
    DocMeta.setdocmeta!(Slanter, :DocTestSetup, :(using Slanter); recursive = true)
    return doctest(Slanter; manual = false)
end

function compute_moment(matrix::AbstractMatrix{<:Real})::AbstractFloat
    n_rows, n_cols = size(matrix)
    row_indices = collect(1:n_rows)
    col_indices = collect(1:n_cols)
    row_indices_matrix = repeat(row_indices, 1, n_cols)
    col_indices_matrix = transpose(repeat(col_indices, 1, n_rows))
    rows_per_col = n_rows / n_cols
    cols_per_row = n_cols / n_rows
    ideal_row_indices_matrix = col_indices_matrix .* rows_per_col
    ideal_col_indices_matrix = row_indices_matrix .* cols_per_row
    row_distance_matrix = row_indices_matrix .- ideal_row_indices_matrix
    col_distance_matrix = col_indices_matrix .- ideal_col_indices_matrix
    distance_matrix = sqrt.(row_distance_matrix .^ 2 .+ col_indices_matrix .^ 2)
    return sum(matrix .* distance_matrix)
end

@testset "reorder_rectangle" begin
    raw_data = rand(10, 20)
    raw_moment = compute_moment(raw_data)
    slanted_data = slanted_reorder(raw_data)
    slanted_moment = compute_moment(slanted_data)
    println("slanted_moment: $(slanted_moment)")
    println("raw_moment: $(raw_moment)")
    @test slanted_moment < raw_moment
end

@testset "reorder_square" begin
    raw_data = rand(10, 10)
    raw_moment = compute_moment(raw_data)
    slanted_data = slanted_reorder(raw_data)
    slanted_moment = compute_moment(slanted_data)
    same_data = slanted_reorder(raw_data; same_order = true)
    same_moment = compute_moment(slanted_data)
    println("slanted_moment: $(slanted_moment)")
    println("same_moment: $(slanted_moment)")
    println("raw_moment: $(raw_moment)")
    @test slanted_moment <= same_moment < raw_moment
end

## The distinct labels of the entries, in the order they appear in, which has one entry per label if (and only if) each
## label covers a contiguous range of the order.
function labels_in_order(order::AbstractVector{<:Integer}, label_per_entry::AbstractVector)::Vector
    labels = label_per_entry[order]
    return labels[[true; labels[2:end] .!= labels[1:(end - 1)]]]
end

@testset "grouped_subgroups" begin
    # Two entries in each of six subgroups, three subgroups in each of two groups. The entries of the different
    # subgroups are identical, so clustering them without the groups interleaves the subgroups completely.
    n_subgroups = 6
    n_groups = 2
    values = reshape([Float64(1 + (index - 1) % 2) + 0.01 * rand() for index in 1:(2 * n_subgroups)], 1, :)
    distances = pairwise(Euclidean(), values; dims = 2)

    # The subgroups are numbered in the opposite order of their groups - the first group holds the last subgroups - so
    # that ordering by the subgroups alone would give the wrong answer, and only ordering by the (group, subgroup) pair
    # gives the right one.
    numbered_subgroups = repeat(1:n_subgroups; inner = 2)
    numbered_groups = [subgroup <= 3 ? 2 : 1 for subgroup in numbered_subgroups]
    named_subgroups = ["S$(index)" for index in numbered_subgroups]
    named_groups = ["G$(index)" for index in numbered_groups]

    group_index_per_subgroup = zeros(Int, n_subgroups)
    for (group_index, subgroup_index) in zip(numbered_groups, numbered_subgroups)
        group_index_per_subgroup[subgroup_index] = group_index
    end

    # Without the groups, the subgroups are indeed interleaved, so the tests below are not vacuous.
    @test length(labels_in_order(ehclust(distances).order, numbered_subgroups)) > n_subgroups

    for (groups_name, groups) in (("named", named_groups), ("numbered", numbered_groups)),
        (subgroups_name, subgroups) in (("named", named_subgroups), ("numbered", numbered_subgroups))

        order = ehclust(distances; groups, subgroups).order
        println("$(groups_name) groups, $(subgroups_name) subgroups: $(order)")

        # Each group, and each subgroup, covers a contiguous range of the order.
        @test length(labels_in_order(order, numbered_groups)) == n_groups
        @test length(labels_in_order(order, numbered_subgroups)) == n_subgroups

        # A numbered level is laid out in the order of its numbers; a named one is laid out by the clustering.
        if groups_name == "numbered"
            @test labels_in_order(order, numbered_groups) == collect(1:n_groups)
        end
        if subgroups_name == "numbered"
            for group_index in 1:n_groups
                subgroups_of_group = filter(
                    subgroup -> group_index_per_subgroup[subgroup] == group_index,
                    labels_in_order(order, numbered_subgroups),
                )
                @test issorted(subgroups_of_group)
            end
        end
    end

    @test_throws AssertionError ehclust(distances; subgroups = named_subgroups)
    # A subgroup is nested in its group, so each group may name its own subgroups the same way; these are then two
    # different subgroups rather than one subgroup spanning both groups.
    reused_subgroups = repeat(["A", "B", "C"]; inner = 2, outer = 2)
    order = ehclust(distances; groups = numbered_groups, subgroups = reused_subgroups).order
    @test length(labels_in_order(order, numbered_groups)) == n_groups
    @test length(labels_in_order(order, numbered_subgroups)) == n_subgroups
    @test length(labels_in_order(order, reused_subgroups)) == n_subgroups
end

@testset "reorder_hclust" begin
    n_rows = 10
    n_cols = 20
    raw_data = rand(10, 20)

    rows_distances = pairwise(Euclidean(), raw_data; dims = 1)
    @assert size(rows_distances) == (10, 10)

    cols_distances = pairwise(Euclidean(), raw_data; dims = 2)
    @assert size(cols_distances) == (20, 20)

    rows_hclust = hclust(rows_distances)
    cols_hclust = hclust(cols_distances)

    hclust_data = raw_data[rows_hclust.order, cols_hclust.order]
    hclust_moment = compute_moment(hclust_data)

    rows_order, cols_order = slanted_orders(hclust_data)
    @assert length(rows_order) == n_rows
    @assert length(cols_order) == n_cols

    rows_reorder = reorder_hclust(rows_hclust, rows_order).order
    cols_reorder = reorder_hclust(cols_hclust, cols_order).order

    rclust_data = hclust_data[rows_reorder, cols_reorder]
    rclust_moment = compute_moment(rclust_data)

    println("rclust_moment: $(rclust_moment)")
    println("hclust_moment: $(hclust_moment)")

    @test rclust_moment < hclust_moment
end
