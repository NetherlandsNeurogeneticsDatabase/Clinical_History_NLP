c25 <- c(
  "dodgerblue2", "#E31A1C", # red
  "green4",
  "#6A3D9A", # purple
  "#FF7F00", # orange
  "black", "gold1",
  "skyblue2", "#FB9A99", # lt pink
  "palegreen2",
  "#CAB2D6", # lt purple
  "#FDBF6F", # lt orange
  "gray70", "khaki2",
  "maroon", "orchid1", "deeppink1", "blue1", "steelblue4",
  "darkturquoise", "green1", "yellow4", "yellow3",
  "darkorange4", "brown"
)
# Create a named character vector with diagnosis-color mappings
diagnosis_colors <- c(
    "AD" = "#8dd3c7",#checked
    'PD,AD' =  '#6fa1f7',
    'AD,CA' = '#105eb4',
    'AD,DLB' = "#3e4772",#'#4086db',
    'AD,ENCEPHA,VE' = '#1c9a8a',
    'DEM,ENCEPHA,VE' = '#FF6EB4',
    'DEM,SICC' = '#FF1493',
    'DEM,SICC,AGD' = '#FF69B4',
    'DLB,SICC' ='#B38166',
    "other" = "#FB9A99", 
    "CON" = "#FCCC3A",#checked
     "FTD" = "#fb8072",#checked
    "FTD_undefined" ="#FF0800",# "#ffb19b", 
    'FTD-TDP' = '#B80F0A',
      'FTD,FTD-TDP_undefined' = '#B80F0A',
      'FTD,FTD-TDP-C' ='#fb9a80',
      'FTD,FTD-TDP-A,PROG' = '#420D09',
      'FTD,FTD-TDP-B,C9ORF72' ='#FF0800',
    'FTD,FTD-TDP,MND' = '#933A16',
    'FTD,FTD-FUS' = '#ffb19b',
    'FTD,FTD-TAU,TAU' = '#fa6d5e',
    'FTD,PID' = '#F5761A',#'#f94f49',
    #  "FTD" = "#fb8072",#"#FF7F00", #orange
    # "FTD_undefined" = "#FF7F00", #orange
    # 'FTD,FTD-TDP' = '#FFA833',
    # 'FTD,FTD-TDP_undefined' = '#FFA833',
    # 'FTD,FTD-TDP-C' ='#D66C00',
    # 'FTD,FTD-TDP-A,PROG' = '#FF5733',
    # 'FTD,FTD-TDP-B,C9ORF72' ='#FFCC66',
    # 'FTD,FTD-TDP,MND' = '#FF8533',
    # 'FTD,FTD-FUS' = '#E64D00',
    # 'FTD,FTD-TAU,TAU' = '#FF9966',
    # 'FTD,PID' = '#FF904D',
    'CBD' = "#9E6FC0",
    "MS" = "#FDB462",#checked
    "MS_undefined" = "#FDB462", #checked
    "MS,MS-PP" = "#FF7417", #checked
    "MS,MS-RR" = "#916C20", ##checked 
    "MS,MS-SP" = "#DAA520", #checked 
    "PD" = "#80B1D3",#checked 
    "PSP" = "#FCCDE5",#checked 
    "VD" = "#7F7F7F", #checked 
    "MSA" = "#BCBEC0", #checked 
    "MND" = "#BC80BD",#checked 
    "ALS" = "#BC80BD",#checked
    "MND_other" = "#BEBADA", #checked 
    "DLB" = "#d8cd80",#checked 
    "ATAXIA" = "#CCEBC5", #checked 
    "MDD" = "#FDBF6F",
    "SCZ" = "maroon", 
    "BP" = "khaki2",
    "other_dem" = "#3e4772",#checked 
    "PSYCH" = "#b3de69", #checked 
    "other_psych" = 'orchid1',
#     "PSYCH,ASD"   = '#FF7F50',  # Coral
#     "PSYCH,BP"    = '#9370DB',  # Medium Purple
#     "PSYCH,MDD"   = '#FFA07A',  # Deep Pink
#     "PSYCH,OCD"   = '#FF4500',  # Orange Red
#     "PSYCH,PTSD"  = '#FF69B4',  # Hot Pink
#     "PSYCH,SCZ"   = '#FF8C00'  # Dark Orange
# )
    "PSYCH,ASD"   = '#018749',  # Original Green
    "PSYCH,BP"    = '#39ff14',  # Slightly Darker Green
    "PSYCH,MDD"   = '#d7f09b',  #  Lighter/Brighter Green
    "PSYCH,OCD"   = '#68a53a',  # Darker and Saturated Green
    "PSYCH,PTSD"  = '#8fbc8f',  # HMedium Brightness Green
    "PSYCH,SCZ"   = '#3d6c15'  # Very Dark Green
)



colors <- c(
'Aspecific symptoms' = '#ce6dbd',
'Autonomic dysfunction' = '#b5cf6b',
'Cerebellar & vestibular system dysfunction' = '#6b6ecf',
'Changes in consciousness, awareness & orientation' = '#d6616b',
'Cognitive and memory impairment' = '#e7ba52',
'Signs of (dis)inhibition' = '#bd9e39',
'Disturbances in mood & behaviour' = '#ad494a',
'Extrapyramidal symptoms' = '#9c9ede',
'General decline' = '#a55194',
'Signs of impaired mobility' = '#393b79',
'Motor deficits' = '#5254a3',
'Other signs & symptoms of cortical dysfunction' = '#e7cb94',
'Other psychiatric signs & symptoms' = '#e7969c',
'Sensory deficits' = '#8ca252'
)

colors_domain <- c(
'Cognitive' = '#F3DB8F',
'Motor' = '#5672AB',
'General' = '#A467A1',
'Psychiatric' = '#AB4A4A',
'Sensory/autonomic' = '#42755E'
)

custom_colors_origin <- c(
  'temporal_and_observational' = '#76B993',          # Pastel Green
  'only_temporal' = '#FFD699',      # Pastel Orange
  'only_observational' = '#5A82CA'  # Pastel Blue
)
custom_colors_origin <- c(
  'temporal_and_observational' = '#333333', # Dark Grey
  'only_temporal' = '#808080',             # Grey
  'only_observational' = '#C0C0C0'         # Light Grey
)