"""Explicit sequence splits for every dataset we run onboarding on.

Three splits per dataset, frozen 2026-09-01:

* ``test``   — everything reported in the paper: the full sweep set.
* ``val``    — the fixed-seed development subset, IDENTICAL to what ``--val`` selects at
               runtime (``utils.dataset_sequences``, VAL_SEED=20260609). Frozen here so it
               is visible and greppable.
* ``sanity`` — a handful of cells for fast smoke tests: one cell per runner code path
               (static, dynamic, HO3D, NAVI, GSO synthetic, BOP classic, YCBInEOAT,
               HOT3D). All sanity cells are members of ``val``.

Non-obvious facts encoded here (do not "fix" them):
* HANDAL/HOPE **test** statics are the ``_up`` orientation for every object (that is what
  the full sweeps ran), while **val** statics use a seeded per-object up/down choice —
  so val is deliberately NOT a subset of test for the static arm.
* HO3D **val** comes from the *train* split (full GT available) and is disjoint from the
  13-sequence *evaluation*-split test set. 
* GSO: the 18 models below are everything under ``data/GoogleScannedObjects/models/``.
* LM-O: ``train`` holds exactly these 8 scenes, so val == test there. tless/icbin are
  supported by the runners but not part of the current protocol; add them here if that
  changes.
* HOT3D is enumerated for completeness but is not in the current headline tables.

Cells are addressed as ``<dataset> <sequence>``, the same pair the ``run_*.py`` entry
points take via ``--sequences``.

CLI:  python -m configs.splits --split val                 # all datasets, manifest lines
      python -m configs.splits --split test --datasets handal hope
"""
from __future__ import annotations

SPLIT_NAMES = ('test', 'val', 'sanity')

# --------------------------------------------------------------------------- handal
_HANDAL_OBJECTS = [f'obj_{i:06d}' for i in range(1, 41)]                      # 40 objects
_HANDAL_TEST = ([f'{o}_up' for o in _HANDAL_OBJECTS]
                + [f'{o}_dynamic' for o in _HANDAL_OBJECTS])
_HANDAL_VAL = [
    'obj_000002_down', 'obj_000010_up', 'obj_000011_down', 'obj_000018_down',
    'obj_000019_down', 'obj_000020_down', 'obj_000032_up', 'obj_000033_up',
    'obj_000002_dynamic', 'obj_000010_dynamic', 'obj_000011_dynamic', 'obj_000018_dynamic',
    'obj_000019_dynamic', 'obj_000020_dynamic', 'obj_000032_dynamic', 'obj_000033_dynamic',
]

# ----------------------------------------------------------------------------- hope
_HOPE_OBJECTS = [f'obj_{i:06d}' for i in range(1, 29)]                        # 28 objects
_HOPE_TEST = ([f'{o}_up' for o in _HOPE_OBJECTS]
              + [f'{o}_dynamic' for o in _HOPE_OBJECTS])
_HOPE_VAL = [
    'obj_000005_down', 'obj_000006_up', 'obj_000009_down', 'obj_000010_up',
    'obj_000016_up', 'obj_000017_up', 'obj_000018_down', 'obj_000024_up',
    'obj_000005_dynamic', 'obj_000006_dynamic', 'obj_000009_dynamic', 'obj_000010_dynamic',
    'obj_000016_dynamic', 'obj_000017_dynamic', 'obj_000018_dynamic', 'obj_000024_dynamic',
]

# ----------------------------------------------------------------------------- ho3d
# test = the full HO3D *evaluation* split; val = 8 seeded *train*-split sequences
# (train has full GT; the two are disjoint on purpose — see module docstring).
_HO3D_TEST = ['AP10', 'AP11', 'AP12', 'AP13', 'AP14',
              'MPM10', 'MPM11', 'MPM12', 'MPM13', 'MPM14',
              'SB11', 'SB13', 'SM1']
_HO3D_VAL = ['BB14', 'GPMF10', 'GSF12', 'GSF14', 'SB10', 'SB12', 'ShSu12', 'ShSu13']

# ----------------------------------------------------------------------------- navi
# All 136 video sequences of navi_v1.5 (full disk enumeration, 'object@video-folder').
_NAVI_TEST = [
    '3d_dollhouse_sink@video-00-canon_t4i-MVI_2649',
    '3d_dollhouse_sink@video-00-pixel_4xl-PXL_20230305_231812384',
    '3d_dollhouse_sink@video-01-canon_t4i-MVI_2682',
    '3d_dollhouse_sink@video-01-pixel_5-PXL_20230305_232742533',
    '3d_dollhouse_sink@video-02-canon_t4i-MVI_2830',
    '3d_dollhouse_sink@video-02-pixel_7-PXL_20230402_064911006.TS',
    '3d_dollhouse_sink@video-03-canon_t4i-MVI_3299',
    '3d_dollhouse_sink@video-03-pixel_7-PXL_20230408_004013979.TS',
    '3d_dollhouse_sink@video-04-canon_t4i-MVI_3508',
    '3d_dollhouse_sink@video-04-pixel_7-PXL_20230416_000227859.TS',
    '3d_dollhouse_sink@video-05-pixel_7-PXL_20230728_004833427.TS',
    'box_multi_door_colored@video-00-pixel_6pro-PXL_20230813_235911646.TS',
    'box_multi_door_colored@video-01-pixel_6pro-PXL_20231019_022623096.TS',
    'bunny_racer@video-00-pixel_5-PXL_20230227_054733115',
    'bunny_racer@video-01-canon_t4i-MVI_2611',
    'bunny_racer@video-01-pixel_5-PXL_20230305_185250184',
    'bunny_racer@video-02-canon_t4i-MVI_2794',
    'bunny_racer@video-02-pixel_7-PXL_20230402_064255567.TS',
    'bunny_racer@video-03-canon_t4i-MVI_3254',
    'bunny_racer@video-03-pixel_7-PXL_20230408_003714583.TS',
    'bunny_racer@video-04-canon_t4i-MVI_3464',
    'bunny_racer@video-04-pixel_7-PXL_20230415_235605526.TS',
    'bunny_racer@video-05-pixel_7-PXL_20230728_005055979.TS',
    'can_kernel_corn@video-00-pixel_6pro-PXL_20230813_235210008.TS',
    'can_kernel_corn@video-01-pixel_6pro-PXL_20231019_021227818.TS',
    'chicken_racer@video-00-pixel_4xl-PXL_20230220_003535743',
    'chicken_racer@video-01-pixel_4xl-PXL_20230220_011207427',
    'chicken_racer@video-02-pixel_5-PXL_20230220_012138141',
    'chicken_racer@video-03-pixel_5-PXL_20230220_013128946',
    'chicken_racer@video-04-pixel_4xl-PXL_20230220_075856662',
    'chicken_racer@video-04-pixel_5-PXL_20230220_075605127',
    'circo_fish_toothbrush_holder_14995988@video-00-canon_t4i-MVI_2304',
    'circo_fish_toothbrush_holder_14995988@video-01-canon_t4i-MVI_2305',
    'circo_fish_toothbrush_holder_14995988@video-01-pixel_5-PXL_20230201_010036476',
    'circo_fish_toothbrush_holder_14995988@video-02-canon_t4i-MVI_2338',
    'circo_fish_toothbrush_holder_14995988@video-03-pixel_5-PXL_20230201_011752178',
    'circo_fish_toothbrush_holder_14995988@video-04-canon_t4i-MVI_3016',
    'circo_fish_toothbrush_holder_14995988@video-04-pixel_7-PXL_20230408_001216861.TS',
    'circo_fish_toothbrush_holder_14995988@video-05-pixel_6pro-PXL_20230220_202658459',
    'circo_fish_toothbrush_holder_14995988@video-06-pixel_6pro-PXL_20230220_203705510',
    'dino_4@video-00-pixel_5-PXL_20221223_192758883',
    'dino_4@video-01-pixel_5-PXL_20221223_193459244',
    'dino_4@video-02-pixel_5-PXL_20221223_211237658',
    'dino_4@video-02-pixel_5-PXL_20221223_212421253',
    'dino_4@video-03-pixel_4xl-PXL_20230131_075355441',
    'dino_4@video-04-pixel_5-PXL_20230131_080529421',
    'dino_4@video-05-pixel_7-PXL_20230728_005450669.TS',
    'dino_5@video-00-canon_t4i-MVI_2524',
    'dino_5@video-00-pixel_5-PXL_20230305_173319741',
    'dino_5@video-01-pixel_5-PXL_20230301_014721371',
    'dino_5@video-02-canon_t4i-MVI_2926',
    'dino_5@video-02-pixel_7-PXL_20230403_055826997.TS',
    'dino_5@video-03-canon_t4i-MVI_3204',
    'dino_5@video-03-pixel_7-PXL_20230408_002731710.TS',
    'dino_5@video-04-canon_t4i-MVI_3400',
    'dino_5@video-04-pixel_7-PXL_20230414_002428139.TS',
    'dino_5@video-05-pixel_7-PXL_20230728_004955641.TS',
    'duck_bath_yellow_s@video-00-pixel_6pro-PXL_20231019_020940651.TS',
    'fire_engine_toy_red_yellow_s@video-00-pixel_6pro-PXL_20230814_000042701.TS',
    'fire_engine_toy_red_yellow_s@video-01-pixel_6pro-PXL_20231019_022049780.TS',
    'garbage_truck_green_toy_s@video-00-pixel_6pro-PXL_20231019_020748075.TS',
    'hut_mushrooms_showpiece@video-00-pixel_6pro-PXL_20230813_235059948.TS',
    'hut_mushrooms_showpiece@video-01-pixel_6pro-PXL_20231019_021402316.TS',
    'ice_cream_cart_showpiece@video-00-pixel_6pro-PXL_20230814_000239918.TS',
    'ice_cream_cart_showpiece@video-01-pixel_6pro-PXL_20231019_022412278.TS',
    'paper_weight_flowers_showpiece@video-00-pixel_6pro-PXL_20230813_235649822.TS',
    'paper_weight_flowers_showpiece@video-01-pixel_6pro-PXL_20231019_021748381.TS',
    'pumpkin_showpiece_s@video-00-pixel_6pro-PXL_20231019_020847279.TS',
    'remote_control_toy_car_s@video-00-pixel_6pro-PXL_20231019_021038549.TS',
    'schleich_african_black_rhino@video-00-pixel_4xl-PXL_20230227_055633915',
    'schleich_african_black_rhino@video-01-pixel_4xl-PXL_20230227_055233199',
    'schleich_african_black_rhino@video-02-canon_t4i-MVI_2449',
    'schleich_african_black_rhino@video-03-pixel_4xl-PXL_20230306_004740604',
    'schleich_african_black_rhino@video-04-canon_t4i-MVI_2716',
    'schleich_african_black_rhino@video-04-pixel_4xl-PXL_20230305_234616664',
    'schleich_african_black_rhino@video-05-pixel_7-PXL_20230728_005135658.TS',
    'schleich_bald_eagle@video-00-pixel_4xl-PXL_20230227_060616800',
    'schleich_bald_eagle@video-00-pixel_5-PXL_20230227_060801367',
    'schleich_bald_eagle@video-01-canon_t4i-MVI_2486',
    'schleich_bald_eagle@video-02-canon_t4i-MVI_2749',
    'schleich_bald_eagle@video-02-pixel_7-PXL_20230402_063015726.TS',
    'schleich_bald_eagle@video-03-canon_t4i-MVI_2891',
    'schleich_bald_eagle@video-03-pixel_7-PXL_20230402_185725161.TS',
    'schleich_bald_eagle@video-04-canon_t4i-MVI_2969',
    'schleich_bald_eagle@video-04-pixel_7-PXL_20230407_234238019.TS',
    'schleich_bald_eagle@video-05-pixel_7-PXL_20230726_170837568.TS',
    'schleich_bald_eagle@video-06-pixel_7-PXL_20230728_005527642.TS',
    'schleich_hereford_bull@video-00-pixel_4xl-PXL_20230220_003300481',
    'schleich_hereford_bull@video-00-pixel_5-PXL_20230220_003027959',
    'schleich_hereford_bull@video-01-pixel_5-PXL_20230220_004517000',
    'schleich_hereford_bull@video-02-pixel_4xl-PXL_20230220_005555450',
    'schleich_hereford_bull@video-02-pixel_5-PXL_20230220_005319983',
    'schleich_hereford_bull@video-03-pixel_4xl-PXL_20230220_010137607',
    'schleich_hereford_bull@video-04-pixel_5-PXL_20230301_014112282',
    'schleich_hereford_bull@video-05-pixel_7-PXL_20230728_005322735.TS',
    'schleich_lion_action_figure@video-00-pixel_4xl-PXL_20230227_060338263',
    'schleich_lion_action_figure@video-00-pixel_5-PXL_20230227_060131038',
    'schleich_lion_action_figure@video-01-canon_t4i-MVI_2409',
    'schleich_lion_action_figure@video-02-pixel_4xl-PXL_20230306_005539992',
    'schleich_lion_action_figure@video-03-pixel_5-PXL_20230311_013132364',
    'schleich_lion_action_figure@video-04-pixel_7-PXL_20230323_182644860.TS',
    'schleich_lion_action_figure@video-05-pixel_7-PXL_20230728_004915170.TS',
    'schleich_spinosaurus_action_figure@video-00-pixel_5-PXL_20230311_005751825',
    'schleich_spinosaurus_action_figure@video-01-canon_t4i-MVI_3344',
    'schleich_spinosaurus_action_figure@video-01-pixel_7-PXL_20230413_055842196.TS',
    'schleich_spinosaurus_action_figure@video-02-canon_t4i-MVI_3546',
    'schleich_spinosaurus_action_figure@video-02-pixel_7-PXL_20230416_001248100.TS',
    'schleich_spinosaurus_action_figure@video-03-canon_t4i-MVI_3591',
    'schleich_spinosaurus_action_figure@video-04-canon_t4i-MVI_3662',
    'schleich_spinosaurus_action_figure@video-04-pixel_7-PXL_20230416_191731344.TS',
    'schleich_spinosaurus_action_figure@video-05-pixel_7-PXL_20230726_170754754.TS',
    'schleich_spinosaurus_action_figure@video-06-pixel_7-PXL_20230728_005233138.TS',
    'school_bus@video-00-pixel_4xl-PXL_20230227_054318400',
    'school_bus@video-01-pixel_4xl-PXL_20230227_015223912',
    'school_bus@video-02-canon_t4i-MVI_2573',
    'school_bus@video-02-pixel_5-PXL_20230305_184959311',
    'school_bus@video-03-pixel_5-PXL_20230311_012557041',
    'school_bus@video-04-pixel_7-PXL_20230728_005610578.TS',
    'soldier_wood_showpiece@video-00-pixel_6pro-PXL_20230814_000424946.TS',
    'soldier_wood_showpiece@video-01-pixel_6pro-PXL_20231019_022330215.TS',
    'steps_small_showpiece@video-00-pixel_6pro-PXL_20230813_235317845.TS',
    'steps_small_showpiece@video-01-pixel_6pro-PXL_20231019_022244962.TS',
    'tractor_green_showpiece@video-00-pixel_6pro-PXL_20230813_235526726.TS',
    'tractor_green_showpiece@video-01-pixel_6pro-PXL_20231019_022135919.TS',
    'tumbler_air_balloon@video-00-pixel_6pro-PXL_20231019_021701134.TS',
    'water_gun_toy_green@video-00-pixel_6pro-PXL_20231019_022519608.TS',
    'water_gun_toy_white@video-00-pixel_6pro-PXL_20231019_021551163.TS',
    'water_gun_toy_yellow@video-00-pixel_6pro-PXL_20231019_021136215.TS',
    'weisshai_great_white_shark@video-00-pixel_5-PXL_20221223_205803236',
    'weisshai_great_white_shark@video-00-pixel_5-PXL_20221223_210146133',
    'weisshai_great_white_shark@video-01-pixel_5-PXL_20221223_195102582',
    'weisshai_great_white_shark@video-02-pixel_4xl-PXL_20221223_200159193',
    'weisshai_great_white_shark@video-03-pixel_4xl-PXL_20230131_075922751',
    'weisshai_great_white_shark@video-04-pixel_7-PXL_20230728_004729862.TS',
    'welcome_sign_mushrooms@video-00-pixel_6pro-PXL_20231019_022830512.TS',
    'well_with_leaf_roof_showpiece@video-00-pixel_6pro-PXL_20231019_021314335.TS',
]
_NAVI_VAL = [
    '3d_dollhouse_sink@video-03-canon_t4i-MVI_3299',
    'circo_fish_toothbrush_holder_14995988@video-04-pixel_7-PXL_20230408_001216861.TS',
    'dino_4@video-01-pixel_5-PXL_20221223_193459244',
    'dino_4@video-02-pixel_5-PXL_20221223_212421253',
    'schleich_african_black_rhino@video-00-pixel_4xl-PXL_20230227_055633915',
    'schleich_african_black_rhino@video-05-pixel_7-PXL_20230728_005135658.TS',
    'schleich_bald_eagle@video-00-pixel_4xl-PXL_20230227_060616800',
    'schleich_spinosaurus_action_figure@video-04-pixel_7-PXL_20230416_191731344.TS',
    'water_gun_toy_yellow@video-00-pixel_6pro-PXL_20231019_021136215.TS',
    'weisshai_great_white_shark@video-00-pixel_5-PXL_20221223_205803236',
    'weisshai_great_white_shark@video-02-pixel_4xl-PXL_20221223_200159193',
    'weisshai_great_white_shark@video-03-pixel_4xl-PXL_20230131_075922751',
]

# ------------------------------------------------------------------------------ gso
# All 18 models under data/GoogleScannedObjects/models/ (synthetic random-walk renders).
_GSO_TEST = [
    'Gigabyte_GA970AUD3P_10_Motherboard_ATX_Socket_AM3',
    'INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count',
    'Nestl_Skinny_Cow_Heavenly_Crisp_Candy_Bar_Chocolate_Raspberry_6_pack_462_oz_total',
    'Perricone_MD_No_Bronzer_Bronzer',
    'Perricone_MD_Vitamin_C_Ester_Serum',
    'SCHOOL_BUS',
    'STACKING_BEAR',
    'Schleich_Allosaurus',
    'Sootheze_Cold_Therapy_Elephant',
    'Squirrel',
    'TOP_TEN_HI',
    'Tag_Dishtowel_Green',
    'Threshold_Ramekin_White_Porcelain',
    'Threshold_Salad_Plate_Square_Rim_Porcelain',
    'Threshold_Textured_Damask_Bath_Towel_Pink',
    'Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure',
    'Twinlab_Nitric_Fuel',
    'Vtech_Stack_Sing_Rings_636_Months',
]
_GSO_VAL = [
    'Nestl_Skinny_Cow_Heavenly_Crisp_Candy_Bar_Chocolate_Raspberry_6_pack_462_oz_total',
    'Perricone_MD_Vitamin_C_Ester_Serum',
    'SCHOOL_BUS',
    'Sootheze_Cold_Therapy_Elephant',
    'Tag_Dishtowel_Green',
    'Threshold_Salad_Plate_Square_Rim_Porcelain',
    'Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure',
    'Vtech_Stack_Sing_Rings_636_Months',
]

# ------------------------------------------------------------------------------ lmo
# lmo/train holds exactly these 8 scenes, so the seeded val saturates: val == test.
_LMO_TEST = [f'lmo@train@{s}' for s in
             ('000001', '000005', '000006', '000008', '000009', '000010', '000011', '000012')]

# ------------------------------------------------------------------------ ycbineoat
_YCBINEOAT_TEST = [
    'bleach0', 'bleach_hard_00_03_chaitanya',
    'cracker_box_reorient', 'cracker_box_yalehand0',
    'mustard0', 'mustard_easy_00_02',
    'sugar_box1', 'sugar_box_yalehand0',
    'tomato_soup_can_yalehand0',
]
# One clip per YCB object (select_ycbineoat_validation, per-object seed).
_YCBINEOAT_VAL = [
    'bleach_hard_00_03_chaitanya', 'cracker_box_yalehand0',
    'mustard0', 'sugar_box_yalehand0', 'tomato_soup_can_yalehand0',
]

# ---------------------------------------------------------------------------- hot3d
# 33 objects, up+down static + dynamic, identical object sets for aria and quest3.
# Not in the current headline tables; kept complete for when it is.
_HOT3D_OBJECTS = [f'obj_{i:06d}' for i in range(1, 34)]
_HOT3D_TEST = ([f'{o}_up' for o in _HOT3D_OBJECTS] + [f'{o}_down' for o in _HOT3D_OBJECTS]
               + [f'{o}_dynamic' for o in _HOT3D_OBJECTS])
_HOT3D_VAL = [  # static-only
    'obj_000005_down', 'obj_000010_up', 'obj_000011_down', 'obj_000016_up',
    'obj_000017_up', 'obj_000018_down', 'obj_000024_up', 'obj_000029_down',
]

# ============================================================================ SPLITS
SPLITS: dict[str, dict[str, list[str]]] = {
    'handal': {
        'test': _HANDAL_TEST,
        'val': _HANDAL_VAL,
        'sanity': ['obj_000002_down', 'obj_000002_dynamic'],
    },
    'hope': {
        'test': _HOPE_TEST,
        'val': _HOPE_VAL,
        'sanity': ['obj_000005_down', 'obj_000005_dynamic'],
    },
    'ho3d': {
        'test': _HO3D_TEST,
        'val': _HO3D_VAL,
        'sanity': ['BB14'],
    },
    'navi': {
        'test': _NAVI_TEST,
        'val': _NAVI_VAL,
        'sanity': ['dino_4@video-01-pixel_5-PXL_20221223_193459244'],
    },
    'gso': {
        'test': _GSO_TEST,
        'val': _GSO_VAL,
        'sanity': ['SCHOOL_BUS'],
    },
    'lmo': {
        'test': _LMO_TEST,
        'val': list(_LMO_TEST),         # val saturates: only 8 scenes exist
        'sanity': ['lmo@train@000001'],
    },
    'ycbineoat': {
        'test': _YCBINEOAT_TEST,
        'val': _YCBINEOAT_VAL,
        'sanity': ['mustard0'],
    },
    'hot3d_aria': {
        'test': _HOT3D_TEST,
        'val': _HOT3D_VAL,
        'sanity': ['obj_000005_down'],
    },
    'hot3d_quest3': {
        'test': list(_HOT3D_TEST),
        'val': list(_HOT3D_VAL),
        'sanity': ['obj_000005_down'],
    },
}


def get_split(dataset: str, split: str) -> list[str]:
    """Sequences of ``dataset`` in ``split`` ('test' | 'val' | 'sanity')."""
    if dataset not in SPLITS:
        raise KeyError(f"unknown dataset '{dataset}' (have: {', '.join(SPLITS)})")
    if split not in SPLIT_NAMES:
        raise KeyError(f"unknown split '{split}' (have: {', '.join(SPLIT_NAMES)})")
    return list(SPLITS[dataset][split])


def iter_cells(split: str, datasets: list[str] | None = None):
    """Yield ``(dataset, sequence)`` cells of a split, manifest-ordered."""
    for dataset in (datasets or SPLITS):
        for sequence in get_split(dataset, split):
            yield dataset, sequence


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description='Print a "<dataset> <sequence>" cells manifest for a split.')
    parser.add_argument('--split', choices=SPLIT_NAMES, required=True)
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='subset of datasets (default: all)')
    args = parser.parse_args()
    for dataset, sequence in iter_cells(args.split, args.datasets):
        print(dataset, sequence)


if __name__ == '__main__':
    main()
