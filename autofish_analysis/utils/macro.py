

STITCHING_MACRO = """
#@ String type
#@ String order
#@ String grid_size_x
#@ String grid_size_y
#@ String tile_overlap
#@ String first_file_index_i
#@ String image_path
#@ String image_name
#@ String output_textfile_name
#@ String output_path
 run("Grid/Collection stitching",
    "type=" + type +
    "order=" + order +
    " grid_size_x=" + grid_size_x +
    " grid_size_y=" + grid_size_y +
    " tile_overlap=" + tile_overlap +
    " first_file_index_i="+first_file_index_i+
    " directory=" + image_path +
    " file_names=" + image_name + 
    " output_textfile_name="+ output_textfile_name +
    " fusion_method=[Linear Blending]"+
    " regression_threshold=0.30"+
    " max/avg_displacement_threshold=2.50"+
    " absolute_displacement_threshold=3.50"+
    " compute_overlap" +   
    " computation_parameters=[Save memory (but be slower)]"+
    " image_output=[Fuse and display]"+
    " output_directory="+output_path);
"""