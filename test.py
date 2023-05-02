import example as exm

if __name__ == "__main__":
    """
    pre-analysis and clean data
    """
    # save_path = 'data/clean_data_no_latlon.csv'
    # exm.testcleandata(save_path, False)
    #
    # save_path = 'data/df_pos_with_state.csv'
    # exm.testclusterpos(save_path, False)
    #
    # read_path = 'data/df_pos_with_state.csv'
    # save_path = 'data/df_pos_with_state_clean2.csv'
    # exm.testmapprovince(read_path, save_path, False)

    """
    train new model for prediction
    """
    read_df_path = 'data/clean_data_no_latlon.csv'
    read_pos_path = 'data/df_pos_with_state_clean.csv'
    folder_edge_name = 'data/edge_pred_conf'
    folder_degree_name = 'data/degree_pred_conf'
    attr_dim = {
        'num_epochs': 100,
        'lr': 0.1,
        'batch_size': 128,
        'scheduler_step_size': 10,
        'scheduler_gamma': 0.7,
        'print_epoch': True,
        'step_print': 10
    }
    exm.testtrainnewmodel(read_df_path, read_pos_path, folder_degree_name, folder_edge_name, attr_dim)

    """
    prediction of edges how to add new node into the graph with lon, lat and pre-trained model
    """
    # exm_lat = 37.063816
    # exm_lon = 92.803324
    # exm.testpredictedges(exm_lat, exm_lon)

    """
    transfer data into front-end
    """
    # read_node_path = 'data/df_pos_cluster.csv'
    # read_edge_path = 'data/df_edge_cluster_light.csv'
    # save_json_path = 'data/railway_data.json'
    # save_maxclusternum_path = 'data/maxclusternum.json'
    # save_maxedgenum_path = 'data/maxedgenum.json'
    # exm.testsetjson(read_node_path, read_edge_path, save_json_path, save_maxclusternum_path, save_maxedgenum_path, False)

    """
    delete node and choose different map type
    """
    # node_id = 10
    # map_path = 'src/assets/graph.pickle'
    # exm.testdeletenode(node_id, map_path)

    """
    Dijkstra algorithm with shortest path
    """
    # source_node = 1298
    # target_node = 1892
    # map_path = 'src/assets/graph.pickle'
    # exm.testshortestway(source_node, target_node, map_path)