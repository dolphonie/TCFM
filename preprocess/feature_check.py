# Features you currently have (from the previous list)
current_features = [
    "timestamp", "icao24", "latitude", "longitude", "groundspeed", "track", 
    "vertical_rate", "callsign", "onground", "alert", "spi", "squawk", 
    "altitude", "geoaltitude", "serials", "hour", "x", "y", "track_unwrapped", 
    "flight_id", "phase", "cluster", "airtmp_sfc", "airtmp_sig", "albclm_sfc", 
    "albedo_sfc", "canmoi_sfc", "ceilht_sfc", "cldmix_sig", "cmpice_sig", 
    "conpac_sfc", "cvcldf_sig", "dewpnt_sig", "dragen_sfc", "dragmo_sfc", 
    "emixht_sig", "emixmt_sig", "gmrefr_sig", "grdtmp_sfc", "grdwet_sfc", 
    "grpmix_sig", "gwetcl_sfc", "htgrte_sig", "icemix_sig", "lahflx_sfc", 
    "lndsea_sfc", "lonflx_sfc", "lwradh_sig", "lwsfdn_sfc", "ncccon_sig", 
    "nggcon_sig", "niicon_sig", "nrrcon_sig", "nsscon_sig", "pblzht_sfc", 
    "perprs_sig", "pottmp_sig", "qqstar_sfc", "radhtr_sig", "ranmix_sig", 
    "rdarrf_sfc", "rdarrf_sig", "relhum_sig", "roughl_sfc", "seaice_sfc", 
    "seatmp_sfc", "sehflx_sfc", "smixco_sig", "snoeqv_sfc", "snomix_sig", 
    "snowdp_sfc", "soilty_sfc", "solflx_sfc", "soltmp_sfc", "stapac_sfc", 
    "stcldf_sig", "swradh_sig", "swsfdn_sfc", "terrht_sfc", "trpres_sfc", 
    "ttlmix_sig", "ttlpcp_sfc", "ttlprs_sig", "ttlsac_sfc", "ttstar_sfc", 
    "turbke_sig", "uustar_sfc", "uutrue_sig", "uuwind_sig", "veggrn_sfc", 
    "vegtyp_sfc", "visibl_sfc", "vvtrue_sig", "vvwind_sig", "wstres_sfc", 
    "wvapor_sig"
]

# Features you want
wanted_features = [
    ("conpac_sfc", "Accumulated Convective Precipitation"),
    ("stapac_sfc", "Accumulated Stable Precipitation"),
    ("ttlpcp_sfc", "Accumulated Total Precipitation"),
    ("airtmp_sig", "Air Temperature"),
    ("cldmix_sig", "Cloud Mixing Ratio [sigm]"),
    ("cld_mix_zht_000000", "Cloud Mixing Ratio [ht_sfc]"),
    ("grpmix_sig", "Graupel Mixing Ratio"),
    ("icemix_sig", "Ice Mixing Ratio [sigm]"),
    ("icemix_zht_000000", "Ice Mixing Ratio [ht_sfc]"),
    ("cmpice_sig", "Icing potential [sigm]"),
    ("cmpice_zht_000000", "Icing potential [ht_sfc]"),
    ("lndsea_sfc", "Land Sea Table"),
    ("latitude", "Latitude"),
    ("longitude", "Longitude"),
    ("ranmix_sig", "Rain Mixing Ratio [sigm]"),
    ("ranmix_zht_0000000", "Rain Mixing Ratio [ht_sfc]"),
    ("relhum_sig", "Relative Humidity"),
    ("snomix_sig", "Snow Mixing Ratio [sigm]"),
    ("snomix_zht_0000000", "Snow Mixing Ratio [ht_sfc]"),
    ("terrht_sfc", "Terrain Height"),
    ("trpres_sfc", "Terrain Pressure"),
    ("ttlprs_sig", "Total Pressure"),
    ("uutrue_sig", "True U-Velocity Component [sigm]"),
    ("uutrue_zht_000000", "True U-Velocity Component [ht_sfc]"),
    ("vvtrue_sig", "True V-Velocity Component [sigm]"),
    ("vvtrue_zht_000000", "True V-Velocity Component [ht_sfc]"),
    ("turbke_sig", "Turbulent Kinetic Energy [sigm]"),
    ("turbke_zht_000000", "Turbulent Kinetic Energy [ht_sfc]"),
    ("wvapor_sig", "Water Vapor Mixing Ratio")
]

# Categorize features
features_present = []
features_missing = []

for feature, description in wanted_features:
    if feature in current_features:
        features_present.append((feature, description))
    else:
        features_missing.append((feature, description))

print("Features already present:")
for feature, description in features_present:
    print(f"- {feature}: {description}")

print("\nMissing features:")
for feature, description in features_missing:
    print(f"- {feature}: {description}")