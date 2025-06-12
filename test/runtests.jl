using Clustering
using Distances
using Random
using Slanter
using Test

Random.seed!(123456)

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
