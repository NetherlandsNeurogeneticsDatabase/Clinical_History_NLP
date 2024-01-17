preprocess_seurat_object <- function(seurat_object, meta_data) {
    ## calculates the percentage of donors with symptombins related to dementia and adds this information as a new metadata column
    seurat_object[["percent.dementia"]] <- PercentageFeatureSet(seurat_object, pattern = "Dementia")
    seurat_object[["percent.executive_dysfunction"]] <- PercentageFeatureSet(seurat_object, pattern = "Execut")
    seurat_object[["percent.memory"]] <- PercentageFeatureSet(seurat_object, pattern = "Memory")

    ##  assigns the sex information from the external metadata table (meta_data) to the Seurat object
    seurat_object[['sex']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'sex']
    seurat_object[['Age_bin']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'age_bin']
    seurat_object[['Age']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'age_at_death']
    seurat_object[['File']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'file_year']
    # seurat_object[['diagnosis_info']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'diagnosis_info']
    # seurat_object[['wrong_diagnosis']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'wrong_diagnosis']
    seurat_object[['diagnosis']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'neuropathological_diagnosis']
    seurat_object[['simplified_diagnosis']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'simplified_diagnosis']
    # seurat_object[['MS_PRS']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'MS']
    # seurat_object[['AD_PRS']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'AD']
    # seurat_object[['FTD_PRS']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'FTD']
    # seurat_object[['AD_misdiagnosis']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'AD_misdiagnosis']
    # seurat_object[['FTD_misdiagnosis']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'FTD_misdiagnosis']
    # seurat_object[['VD_misdiagnosis']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'DLB_misdiagnosis']
    # seurat_object[['DLB_misdiagnosis']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'AD_misdiagnosis']
    # seurat_object[['ATAXIA_misdiagnosis']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'ATAXIA_misdiagnosis']
    # seurat_object[['PD_misdiagnosis']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'PD_misdiagnosis']
    # seurat_object[['PSP_misdiagnosis']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'PSP_misdiagnosis']
    # seurat_object[['MS_misdiagnosis']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'MS_misdiagnosis']
    # seurat_object[['MSA_misdiagnosis']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'MSA_misdiagnosis']
    # seurat_object[['MND_misdiagnosis']] <- meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'MND_misdiagnosis']

    ### some options for ellips analysis
    # seurat_object[['model_and_clinic1']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'model_and_clinic1']
    # seurat_object[['model_and_clinic2']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'model_and_clinic2']
    # seurat_object[['model_and_clinic3']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'model_and_clinic3']
    # seurat_object[['model_and_clinic4']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'model_and_clinic4']
    # seurat_object[['clin1']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'clin1']
    # seurat_object[['clin2']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'clin2']
    seurat_object[['clin_coherence']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'clin_coherence']
    seurat_object[['APOE']] = meta_data[match(names(seurat_object$orig.ident), rownames(meta_data)),'APOE status']


    
    ## convert diagnosis in one-hot encoded diagnosis (all zero --> AD)
    diagnosis_encoded = stats::model.matrix(~diagnosis, data.table(seurat_object[['diagnosis']]))
    diagnosis_encoded= as.data.frame(diagnosis_encoded)
    head(diagnosis_encoded)

    ## use to appoint specific diagnosis (for quick and easy plots)
    # seurat_object[['diagnosisCON']]= diagnosis_encoded[,'diagnosisCON']
    # seurat_object[['diagnosisPD']]= diagnosis_encoded[,'diagnosisPD']
    # seurat_object[['diagnosisMS']]= diagnosis_encoded[,'diagnosisMS']
    # seurat_object[['diagnosisFTD']]= diagnosis_encoded[,'diagnosisFTD']
    # is_AD <- rowSums(diagnosis_encoded[, -1]) == 0
    # seurat_object[["diagnosisAD"]] <- as.integer(is_AD)
    


    return(seurat_object)
}

calculate_p_values <- function(seurat_object,diagnosis_column) {
  contingency_table <- table(seurat_object@meta.data[[diagnosis_column]], seurat_object@meta.data$seurat_clusters)
  Xsq <- chisq.test(contingency_table)
  p_values_df <- data.frame(matrix(1, nrow = nrow(contingency_table), ncol = ncol(contingency_table)))
  rownames(p_values_df) <- rownames(contingency_table)
  colnames(p_values_df) <- colnames(contingency_table)
  for (foo in rownames(contingency_table)) {
      for (cluster in colnames(contingency_table)) {
          cluster_residual <- Xsq$stdres[foo, cluster]
          expected_count <- Xsq$expected[foo, cluster]
          if (cluster_residual >= 0) {
            chi_square <- (contingency_table[foo, cluster] - expected_count)^2 / expected_count
            p_value <- 1 - pchisq(chi_square, df = 1)
            p_values_df[foo, cluster] <- p_value
              }
          }
      }
  return(p_values_df)
}

plot_cluster_diagnosis <- function(seurat_object,p_values_df,diagnosis_column,color = 'white') {
  library(viridis)  
  library(ggplot2)
  library(dplyr)
  library(RColorBrewer)
  library(pals)
  
  cluster_diagnosis <- data.frame(cluster = Idents(seurat_object), diagnosis = seurat_object[[diagnosis_column]])
  colnames(cluster_diagnosis) <- c("cluster", "diagnosis")
  # cluster_diagnosis  
  cluster_diagnosis_percent <- cluster_diagnosis %>%
    group_by(cluster, diagnosis) %>%
    summarise(percentage = n()) %>%
    group_by(cluster) %>%
    mutate(percentage = percentage / sum(percentage) * 100)
  # cluster_diagnosis_percent
  cluster_counts <- cluster_diagnosis %>%
    group_by(cluster) %>%
    summarise(cluster_count = n()) %>%
    group_by(cluster) %>%
    mutate(cluster_count = sum(cluster_count))
  
  cluster_diagnosis_percent <- as.data.frame(cluster_diagnosis_percent)
  p_values <- vector()
  for (i in 1:nrow(cluster_diagnosis_percent)) {
    cluster_val <- as.character(cluster_diagnosis_percent$cluster[i])
    diagnosis_val <- as.character(cluster_diagnosis_percent$diagnosis[i])
    p_value <- p_values_df[diagnosis_val, cluster_val]
    p_values <- c(p_values, p_value)
  }
  cluster_diagnosis_percent$p_value <- p_values  
  cluster_diagnosis_percent$significance <- ifelse(cluster_diagnosis_percent$p_value <= 0.000005, "***",
                                                   ifelse(cluster_diagnosis_percent$p_value <= 0.0005, "**",
                                                          ifelse(cluster_diagnosis_percent$p_value <= 0.05, "*", "")))
  # print(cluster_diagnosis_percent)
  
  plot <- ggplot(cluster_diagnosis_percent, aes(x = factor(cluster), y = percentage, fill = diagnosis,width = 0.4)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = significance), size = 10, position = position_stack(vjust = 0.2), color = color) +
    geom_text(aes(cluster, 105, label = cluster_count, fill = NULL),size = 5, data = cluster_counts) +
    labs(x = "UMAP Cluster", y = "Percentage") +
    scale_fill_manual(name = "Diagnosis", values = diagnosis_colors) +
    theme_minimal() +
    theme(
      legend.text = element_text(size = 20),  # Increase legend font size
      legend.title = element_text(size = 20),
      axis.title = element_text(size = 18),  # Increase axis title font size
      axis.text = element_text(size = 16),
      plot.margin = margin(t = 20, r = 20, b = 50, l = 20, unit = "pt")  # Adjust bottom margin
    )

  annotate_text <- annotate("text", x = 4.5, y = -5, label = "* <= 0.05 | ** <= 0.005 | *** <= 0.0005", size = 5, hjust = 0)

  plot + annotate_text
  return(plot)
}

plot_cluster_diagnosis_diag <- function(seurat_object,p_values_df,diagnosis_column,color = 'white',p_values_df_noncoherent) {
  library(viridis)  
  library(ggplot2)
  library(dplyr)
  library(RColorBrewer)
  library(pals)
  
  cluster_diagnosis <- data.frame(cluster = Idents(seurat_object), diagnosis = seurat_object[[diagnosis_column]])
  colnames(cluster_diagnosis) <- c("cluster", "diagnosis")
  # cluster_diagnosis  
  cluster_diagnosis_percent <- cluster_diagnosis %>%
    group_by(cluster, diagnosis) %>%
    summarise(percentage = n()) %>%
    group_by(cluster) %>%
    mutate(percentage = percentage / sum(percentage) * 100)
  # cluster_diagnosis_percent
  cluster_counts <- cluster_diagnosis %>%
    group_by(cluster) %>%
    summarise(cluster_count = n()) %>%
    group_by(cluster) %>%
    mutate(cluster_count = sum(cluster_count))
   
  
  cluster_diagnosis_percent <- as.data.frame(cluster_diagnosis_percent)
  # print(cluster_diagnosis_percent)   
  p_values <- vector()
  for (i in 1:nrow(cluster_diagnosis_percent)) {
    cluster_val <- as.character(cluster_diagnosis_percent$cluster[i])
    diagnosis_val <- as.character(cluster_diagnosis_percent$diagnosis[i])
    p_value <- p_values_df[diagnosis_val, cluster_val]
    p_values <- c(p_values, p_value)
  }
  p_values_noncoherent <- vector()
  for (i in 1:nrow(cluster_diagnosis_percent)) {
    cluster_val <- as.character(cluster_diagnosis_percent$cluster[i])
    diagnosis_val <- as.character(cluster_diagnosis_percent$diagnosis[i])
    p_value_noncoherent <- p_values_df_noncoherent[diagnosis_val, cluster_val]
    p_values_noncoherent <- c(p_values_noncoherent, p_value_noncoherent)
  }
  # p_values  
  cluster_diagnosis_percent$p_value <- p_values
  cluster_diagnosis_percent$p_value_non_coherent <- p_values_noncoherent  
  cluster_diagnosis_percent$significance <- ifelse(cluster_diagnosis_percent$p_value <= 0.000005, "***",
                                                   ifelse(cluster_diagnosis_percent$p_value <= 0.0005, "**",
                                                          ifelse(cluster_diagnosis_percent$p_value <= 0.05, "*", "")))
  cluster_diagnosis_percent$significance_non_coherent <- ifelse(cluster_diagnosis_percent$p_value_non_coherent <= 0.000005, "***",
                                                                ifelse(cluster_diagnosis_percent$p_value_non_coherent <= 0.0005, "**",
                                                                       ifelse(cluster_diagnosis_percent$p_value_non_coherent <= 0.05, "*", 
                                                                              ifelse(cluster_diagnosis_percent$p_value_non_coherent <= 0.1, "+", ""))))
  # print(cluster_diagnosis_percent)
  
  plot <- ggplot(cluster_diagnosis_percent, aes(x = factor(cluster), y = percentage, 
                                                width = 0.7,
                                                fill = diagnosis)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = significance), size = 8, position = position_stack(vjust = 0.5), color = color) +
    geom_text(aes(label = significance_non_coherent), size = 8, position = position_stack(vjust = 0.25), color = 'black') +
    geom_text(aes(cluster, 105, label = cluster_count, fill = NULL),size = 8, data = cluster_counts) +
    labs(x = "UMAP Cluster", y = "Percentage") +
    scale_fill_manual(name = "Diagnosis", values = diagnosis_colors) +
    theme_minimal() +
    theme(
      legend.text = element_text(size = 20),  # Increase legend font size
      legend.title = element_text(size = 20),
      axis.title = element_text(size = 18),  # Increase axis title font size
      axis.text = element_text(size = 16),
      plot.margin = margin(t = 20, r = 20, b = 50, l = 20, unit = "pt")  # Adjust bottom margin
    )

  annotate_text <- annotate("text", x = 4.5, y = -5, label = "* <= 0.05 | ** <= 0.005 | *** <= 0.0005", size = 5, hjust = 0)

  plot + annotate_text
  return(plot)
}

## advanced dimplot to vizualize umap with circles
generate_dim_plot <- function(seurat_object,unique_diagnoses,diag='simplified_diagnosis',p3) {
    # options(repr.plot.width = 16, repr.plot.height = 10)
    plot_list <- list()
    outside_donor_ids <- c()
    for (di in unique_diagnoses) {
      umap_coords <- Embeddings(seurat_object, "wnn.umap")
      ms_umap_coords <- umap_coords[seurat_object[[diag]] == di, ]
      center <- colMeans(ms_umap_coords)
      distances <- sqrt(rowSums((ms_umap_coords - center)^2))
      num_donors <- nrow(ms_umap_coords)

      desired_percentage_inside <- 0.95
      wanted_outside <- round(num_donors - (num_donors * desired_percentage_inside)) 
      current_radius_x <- 0.5
      current_radius_y <- 0.5
      outside_ids <- c()   

      while (TRUE) {
        all_x_distances <- abs(ms_umap_coords[, 1] - center[1]) 
        current_x_outside <- sum(all_x_distances > current_radius_x) 
        if (current_x_outside <= wanted_outside) {
          break
        }
        current_radius_x <- current_radius_x + 0.05  # Increase the radius incrementally
      }
      while (TRUE) {
        all_y_distances <- abs(ms_umap_coords[, 2] - center[2]) 
        current_y_outside <- sum(all_y_distances > current_radius_y) 
        if (current_y_outside <= wanted_outside) {
          break
        }
        current_radius_y <- current_radius_y + 0.05  # Increase the radius incrementally
      }

      circle_color <- diagnosis_colors[di] 
      ellipse_data <- data.frame(x = center[1], y = center[2], x_radius = current_radius_x, y_radius = current_radius_y)
      distances <- sqrt(((ms_umap_coords[, 1] - ellipse_data$x) / ellipse_data$x_radius)^2 +
                       ((ms_umap_coords[, 2] - ellipse_data$y) / ellipse_data$y_radius)^2)

      outside_ids <- rownames(ms_umap_coords)[distances > 1]
      outside_donor_ids[[di]] <- outside_ids  
      ms_umap_coords_df <- as.data.frame(ms_umap_coords)  
      outside_data <- ms_umap_coords_df %>% filter(rownames(ms_umap_coords_df) %in% outside_ids)  
      p3 <- p3 + geom_ellipse(data = ellipse_data, aes(x0 = x, y0 = y, a = x_radius, b = y_radius, angle = 0),
                             color = circle_color, fill = NA, linetype = "dashed", size = 1.5)#+
    # geom_text(data = outside_data,
    #           aes(x = wnnUMAP_1, y = wnnUMAP_2, label = 'j'),nudge_x = 0.1, nudge_y = 0.1, size = 3, color = "red")

    }
    return(list(outside_donor_ids,p3))
}

find_extended_cluster_markers <- function(seurat_object, n, method = "topn", assay = 'FLAT', verbose = FALSE) {
  extended_cluster_markers <- list()
  for (cluster_number in unique(Idents(seurat_object))) {
    cluster_markers <- FindMarkers(seurat_object, ident.1 = cluster_number, min.pct = 0.1, assay = assay)
    if (verbose) {
      print(cluster_number)
    }
    if (method == "topn") {
      cluster_markers <- head(cluster_markers %>% arrange(desc(p_val_adj)), n)
    } else if (method == "all") {
      # cluster_markers <- cluster_markers %>% filter(p_val_adj < n)
      markers_info <- data.frame(
          cluster = cluster_number,
          marker = rownames(cluster_markers),
          p_val = cluster_markers$p_val,
          avg_log2FC = cluster_markers$avg_log2FC,
          pct.1 = cluster_markers$pct.1,
          pct.2 = cluster_markers$pct.2,
          p_val_adj = cluster_markers$p_val_adj
      )
      extended_cluster_markers[[cluster_number]]   <- markers_info
      if (verbose) {
        # print(head(cluster_markers, 5))
        print(head(markers_info))
        # print(head(extended_cluster_markers))  
      }
    } else {
      stop("Invalid method specified. Choose either 'topn' or 'sign'.")
    }
  }
  
  extended_cluster_markers_df <- do.call(rbind, extended_cluster_markers)
  return(extended_cluster_markers_df)
}

## return significant markers
create_marker_df <- function(temp, flat, sup3) {
    print(dim(flat))
    ## filter only significant ones
    temp <- temp[temp$p_val_adj < 0.05, ]
    flat <- flat[flat$p_val_adj < 0.05, ]
    print(dim(flat))
    marker_list_temp <- temp %>%
      mutate(marker = gsub("\\.|\\d", "", marker)) %>%
      pull(marker)
    marker_list_temp <- unique(marker_list_temp)
    marker_list_flat <- flat %>%
      pull(marker)
    comlist <- union(marker_list_temp, marker_list_flat) ## this is needed

    temporal_and_observational <- intersect(marker_list_temp, marker_list_flat)
    only_temporal <- setdiff(marker_list_temp, temporal_and_observational)
    only_observational <- setdiff(marker_list_flat, temporal_and_observational)
    marker_dictionary <- list(
      temporal_and_observational = temporal_and_observational,
      only_temporal = only_temporal,
      only_observational = only_observational
    )
    update_values <- function(values, lookup_df) {
      new_values <- lookup_df$AttributeUpdated[match(values, lookup_df$ITname)]
      new_values[is.na(new_values)] <- values[is.na(new_values)]
      return(new_values)
    }
    sup3_small <- sup3[c("ITname","AttributeUpdated", "Domain", "Grouping")]
    marker_dictionary$temporal_and_observational <- update_values(marker_dictionary$temporal_and_observational, sup3_small)
    marker_dictionary$only_temporal <- update_values(marker_dictionary$only_temporal, sup3_small)
    marker_dictionary$only_observational <- update_values(marker_dictionary$only_observational, sup3_small)
    marker_df <- data.frame(
      marker = unlist(marker_dictionary),
      origin = rep(names(marker_dictionary), sapply(marker_dictionary, length))
    )
    marker_df$origin <- rownames(marker_df)
    marker_df$origin <- sub("\\d+$", "", marker_df$origin)
    rownames(marker_df) <- NULL
    anndf <- data.frame(features = marker_df$marker,origin=marker_df$origin)
    anndf <- merge(anndf, sup3_small, by.x = "features", by.y = "AttributeUpdated", all.x = TRUE)
    anndf <- anndf[order(anndf$Domain, anndf$Grouping, anndf$features), ]
    return(list(comlist, anndf))
}

## heatmap of significant markers
generate_heatmap_plot <- function(seurat_object,assay, extended_cluster_markers, annotation_df,breaksmax,
                                  savepath="/home/jupyter-n.mekkes@gmail.com-f6d87/clinical_history/final_predictions/figures/seurat/main/heatmap_main_cluster.pdf") {
  options(repr.plot.width = w, repr.plot.height = h)  
  library(pheatmap)
    
  avgexp <- AverageExpression(seurat_object, features = extended_cluster_markers,assays = assay)
  heatmap_data <- as.data.frame(avgexp)
  matrix_data <- data.matrix(heatmap_data)
  
  original_rownames <- rownames(matrix_data)
  new_rownames <- annotation_df$features[match(original_rownames, annotation_df$ITname)]
  new_rownames[is.na(new_rownames)] <- original_rownames[is.na(new_rownames)]
  rownames(matrix_data) <- new_rownames  
  custom_levels_domain <- c('General', 'Motor', 'Sensory/autonomic', 'Cognitive', 'Psychiatric')
  custom_levels_grouping <- c(
      'Aspecific symptoms', 'General decline', 'Extrapyramidal symptoms',
      'Cerebellar & vestibular system dysfunction', 'Motor deficits',
      'Signs of impaired mobility', 'Autonomic dysfunction',
      'Sensory deficits', 'Other signs & symptoms of cortical dysfunction',
      'Cognitive and memory impairment', 'Signs of (dis)inhibition',
      'Other psychiatric signs & symptoms',
      'Changes in consciousness, awareness & orientation',
      'Disturbances in mood & behaviour'
    )
  annotation_df <- annotation_df %>%
    mutate(
        Domain = factor(Domain, levels = custom_levels_domain),
        Grouping = factor(Grouping, levels = custom_levels_grouping)
    ) %>%
    arrange(Domain, Grouping, features)
  # print(annotation_df)  
  ordered_features <- annotation_df$features
  order_indices <- match(ordered_features, rownames(matrix_data))
  matrix_data <- matrix_data[order_indices, ]
  annotation_row <- data.frame(Significance=annotation_df$origin,Domain = annotation_df$Domain)#Grouping = annotation_df$Grouping, ,
  # annotation_row$Origin <- factor(annotation_row$Origin, levels = names(custom_colors_origin))  
  rownames(annotation_row) <- annotation_df$features
  # annotation_colors <- colors[unique(annotation_row$Grouping)]
  annotation_colors_domain <- colors_domain[unique(annotation_row$Domain)]
  # print(unique(annotation_row$Origin))
  print(unique(annotation_row$Domain))
  print(colors_domain)  
  col <- viridis(8)
  annotation_colors_origin <- custom_colors_origin[unique(annotation_row$Significance)]
  breaks <- seq(0, breaksmax, length.out = 11)
  quantile_breaks <- function(xs, n = 10) {
    breaks <- quantile(xs, probs = seq(0, 1, length.out = n))
    breaks[!duplicated(breaks)]
  }
  # breaks <- quantile_breaks(matrix_data, n = 12)
  breaks=c(0,0.25,0.5,1,2,3,4,6,10)
  print(breaks)  
  heatmap_plot <- pheatmap(
    matrix_data, scale = "none", col = col, cluster_rows = FALSE, cluster_cols = FALSE,
      
    breaks = breaks,
    fontsize = 12,annotation_row = annotation_row,
      annotation_colors = list(Significance = annotation_colors_origin,Domain=colors_domain) #Grouping = annotation_colors,,
  )
  heatmap_plot
  ggsave(savepath, plot = heatmap_plot, width = w, height = h,dpi = 300)  
  return(rownames(matrix_data))
}

perform_pca_and_visualization <- function(seurat_object,rname='pca') {
## features = VariableFeatures(object = seurat_object) specifies that you want to use 
## the variable features identified earlier (using FindVariableFeatures) for PCA.
## This ensures that only the most informative features are included in the PCA analysis, 
## which can help reduce noise and focus on the most relevant variation in the data.
    seurat_object <- RunPCA(object = seurat_object,
                            verbose = FALSE,
                            features = VariableFeatures(object = seurat_object),
                            npcs = 50, # default is 50
                            reduction.name = rname
                )

    # Examine and visualize PCA results a few different ways
    print(seurat_object[[rname]], dims = 1:5, nfeatures = 5)
    options(repr.plot.width = 12, repr.plot.height = 8)
    print(VizDimLoadings(seurat_object, dims = 1:3, reduction = rname))
    print(DimPlot(seurat_object, reduction = rname,dims=c(1,2),group.by = "diagnosis"))
    print(DimPlot(seurat_object, reduction = rname,dims=c(1,2),group.by = "simplified_diagnosis"))
    ## JackStraw procedure assesses the statistical significance of each principal component 
    ## by permuting the associations between symptombins and donors 
    ## It generates p-values for each principal component, indicating the likelihood of observing similar or stronger associations by chance.
    # seurat_object <- JackStraw(seurat_object, num.replicate = 100)
    # seurat_object <- ScoreJackStraw(seurat_object, dims = 1:20)
    # ## helps visualize the strength of associations between the principal components and features
    # print(JackStrawPlot(seurat_object, dims = 1:20))
    # ## displays the cumulative variance explained by each principal component. 
    # print(ElbowPlot(seurat_object))
    return(seurat_object)
}

## returns an overview of inside/outside circle metrics
coherent_analysis <- function(unique_diagnoses, variables_to_loop, seurat_object,out_ids) {
  results_list <- list()

  for (di in unique_diagnoses) {
    for (var in variables_to_loop) {
      # print(di)  
      outside_circle <- data.frame(DonorID = rownames(seurat_object@meta.data),
                                   seurat_clusters = seurat_object@meta.data$seurat_clusters,
                                   simplified_diagnosis = seurat_object@meta.data$simplified_diagnosis,
                                   diag = seurat_object@meta.data[[var]])
      order_levels <- c("coherent","non-coherent")
      outside_circle$diag <- factor(outside_circle$diag, levels = order_levels)
      
      outside_circle <- outside_circle[outside_circle$simplified_diagnosis == di, ]
      outside_donor_di <- out_ids[[di]]
      
      outside_circle <- outside_circle %>%
        mutate(location = ifelse(DonorID %in% outside_donor_di, "outside", "inside"))
      contingency_table <- table(outside_circle$diag, outside_circle$location)
      # print(contingency_table)  
      if (dim(contingency_table)[2] < 2 || dim(contingency_table)[1] <2) {
          print(paste("Skipping", var, "due to 1 column contingency_table"))
          next  # Skip to the next var iteration
      }
      fisher_result <- fisher.test(contingency_table)
      if (any(contingency_table == 0)){
          print(paste("Skipping chi for", var, "due to 0 in contingency_table"))
          chi_square_result$p.value <- NaN   
          chi_square_result$statistic <- NaN  
          
      } else {
          chi_square_result <- chisq.test(contingency_table)
          
      }
      
      result <- outside_circle %>%
        group_by(location, diag) %>%
        summarise(num_donors = n(), .groups = "drop") %>%
        ungroup() %>%
        group_by(location) %>%
        mutate(total_num_donors = sum(num_donors)) %>%
        mutate(percentages = (num_donors / total_num_donors) * 100)
      
      result$diagnosis <- di
      result$method <- var
      result$chi.pvalue <- chi_square_result$p.value
      result$chi.stat <- chi_square_result$statistic
      result$fish.pvalue <- fisher_result$p.value       
      result <- result %>%
        filter(!(location == 'outside'  & diag == "coherent") &
               !(location == 'inside' & diag == "non-coherent"))
      # print(colnames(result))
      inside_coh_perc <- subset(result, location == "inside")$percentages
      outside_non_coh_perc <- subset(result, location == "outside")$percentages
      average_percentage <- mean(result$percentages)
      coh_inside_effect <- (inside_coh_perc - 50) * 0.02
      non_coh_outside_effect <- (outside_non_coh_perc - 50) * 0.02
      average_effect <- (average_percentage - 50) * 0.02
      
      result$coh_inside_effect <- coh_inside_effect
      result$non_coh_outside_effect <- non_coh_outside_effect
      result$average_effect <- average_effect
      results_list[[paste(di, var, sep = "_")]] <- result
    }
  }

  combined_results <- bind_rows(results_list)
  combined_results <- unique(combined_results[, c("diagnosis", "method", "chi.pvalue", "chi.stat","fish.pvalue",
                                                  "coh_inside_effect", "non_coh_outside_effect", "average_effect")])

  combined_results$adjusted_pvalue_chi <- p.adjust(combined_results$chi.pvalue, method = "BH")
  combined_results$adjusted_pvalue_fish <- p.adjust(combined_results$fish.pvalue, method = "BH")
  
  return(combined_results)
}

calculate_p_values_diag <- function(seurat_object,diagnosis_column,variable_to_loop='clin2',rows_below_threshold,known_clin_diag,verbose=TRUE) {
    df_pd <- data.frame(DonorID = rownames(seurat_object@meta.data),
                        seurat_clusters = seurat_object@meta.data$seurat_clusters,
                        diagnosis = seurat_object@meta.data[[diagnosis_column]],
                        quality =   seurat_object@meta.data[[variable_to_loop]]
    )    
    order_levels <- c("unknown/coherent","non_coherent")
    df_pd$quality <- factor(df_pd$quality, levels = order_levels)
    if (verbose == TRUE){
        print(table(df_pd$quality))
        print(table(df_pd$seurat_clusters))
    }
    p_values <- data.frame(Diagnosis = character(), Cluster = integer(), P_Value = numeric())
    p_values_diag <- data.frame(Diagnosis = character(), Cluster = integer(), P_Value = numeric())
    # print(unique(df_pd$diagnosis))
    # un <- c('PD')
    # rows_below_threshold <- c('PD','MSA','PSP')
    combined_p_values_df <- data.frame()
    for (di in rows_below_threshold){
        for (var in unique(variable_to_loop)){
            
            temp <- df_pd[df_pd$diagnosis == di, ]
            contingency_table_diag <- table(temp$quality, temp$seurat_clusters)
            if (verbose == TRUE){
                print(di)
                print(table(temp$seurat_clusters))
                print(contingency_table_diag)
            }
            ## other approach for diagnosis overrepresentation
              for (cluster in unique(df_pd$seurat_clusters)) {
                count_di_clusters <- sum(df_pd$diagnosis == di & df_pd$seurat_clusters == cluster)   
                count_notdi_clusters <- sum(df_pd$diagnosis != di & df_pd$seurat_clusters == cluster) 
                count_di_otherclusters <- sum(df_pd$diagnosis == di & df_pd$seurat_clusters != cluster)    
                count_notdi_otherclusters <- sum(df_pd$diagnosis != di & df_pd$seurat_clusters != cluster)
                if (verbose == TRUE) {
                    print(count_di_clusters)
                    print(count_notdi_clusters)  
                    print(count_di_otherclusters)  
                    print(count_notdi_otherclusters)
                }
                  
                contingency_table <- matrix(c(count_di_clusters, count_di_otherclusters, count_notdi_clusters, count_notdi_otherclusters), nrow = 2)
                # print(contingency_table)  
                if (nrow(contingency_table) >= 2 && ncol(contingency_table) >= 2) {
                  # Perform Fisher's exact test (you can use chi-squared test too)
                  test_result <- fisher.test(contingency_table, alternative = "greater")
                  p_values <- rbind(p_values, data.frame(Diagnosis = di, Cluster = cluster, P_Value = test_result$p.value))
                   
                }
                # print(p_values)
                if (di %in% known_clin_diag) {
                    count_noncoh_di_cluster <- sum(df_pd$diagnosis == di & df_pd$seurat_clusters == cluster & df_pd$quality =='non_coherent')  
                    count_coh_di_cluster <- sum(df_pd$diagnosis == di & df_pd$seurat_clusters == cluster & df_pd$quality =='unknown/coherent')
                    count_noncoh_di_othercluster <- sum(df_pd$diagnosis == di & df_pd$seurat_clusters != cluster & df_pd$quality =='non_coherent')
                    count_coh_di_othercluster <- sum(df_pd$diagnosis == di & df_pd$seurat_clusters != cluster & df_pd$quality =='unknown/coherent')  
                    # print(count_noncoh_di_cluster) 
                    # print(count_coh_di_cluster)
                    # print(count_noncoh_di_othercluster)
                    # print(count_coh_di_othercluster)  
                    contingency_table_diag_cluster <- matrix(c(count_noncoh_di_cluster, count_noncoh_di_othercluster, count_coh_di_cluster, count_coh_di_othercluster), nrow = 2)
                    if (nrow(contingency_table_diag_cluster) >= 2 && ncol(contingency_table) >= 2) {
                      test_result_diag <- fisher.test(contingency_table_diag_cluster, alternative = "greater")
                      p_values_diag <- rbind(p_values_diag, data.frame(Diagnosis = di, Cluster = cluster, P_Value = test_result_diag$p.value))
                    }  
                 # print(p_values_diag)   
                }    
              }
    
            
            
            
#             if (dim(contingency_table)[2] < 2 || dim(contingency_table)[1] <2) {
#                 print(paste("Skipping", var, "due to 1 column contingency_table"))
#                 next  # Skip to the next var iteration
#             }
#             if (all(contingency_table == 0)){
#                 print(paste("Skipping chi for", var, "due to 0 in contingency_table"))
#                 next#chi_square_result$p.value <- NaN   
                
#             } else {
#                 Xsq <- chisq.test(contingency_table)
#             }
            
#             p_values_df <- data.frame(matrix(9, nrow = nrow(contingency_table), ncol = ncol(contingency_table)))
#             rownames(p_values_df) <- rownames(contingency_table)
#             colnames(p_values_df) <- colnames(contingency_table) ## columns are clusters, rows are coherent vs non-coherent
#             # print(p_values_df)
#             if (verbose == TRUE){
#                 print(Xsq$expected)
#             }
            
#             for (coh_noncoh in rownames(contingency_table)) {
#               for (cluster in colnames(contingency_table)) {
#                   cluster_residual <- Xsq$stdres[coh_noncoh, cluster] ## measure of differece. > 0: more than expected. <0 less than expected
#                   expected_count <- Xsq$expected[coh_noncoh, cluster]
#                   # print(coh_noncoh)
#                   # print(cluster_residual)
                  
                  
#                   if (!is.na(cluster_residual) && cluster_residual >= 0 && coh_noncoh == 'non_coherent') { ## only the cases where there are more than expected and only for the non-coherent 
#                     chi_square <- (contingency_table[coh_noncoh, cluster] - expected_count)^2 / expected_count
#                     p_value <- 1 - pchisq(chi_square, df = 1)
#                     p_values_df[coh_noncoh, cluster] <- p_value
#                       }
#                   }
#               }
#             p_values_df <- p_values_df[-which(rownames(p_values_df) == "unknown/coherent"), ]
#             rownames(p_values_df)[rownames(p_values_df) == "non_coherent"] <- di
#             combined_p_values_df <- rbind(combined_p_values_df, p_values_df)
            # print(p_values_df)
        }    


        
    }
    if (verbose == TRUE){
        print(p_values)
        print(p_values_diag)
    }
    
    return(list(p_values = p_values, p_values_diag = p_values_diag))
    # return(combined_p_values_df)
}


      # fisher_result <- fisher.test(contingency_table)

