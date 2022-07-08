###
### Optimise roadmap based on client growth or arr growth
###

from functions import *
from settings import *

def main(df_ref, OPTIMISATION_SEGMENT, SCHEMES):
    ###
    ### SETTINGS
    ###
    print('\n', get_time(settings.time0), 'SETTINGS:', '\n')
    print(get_time(settings.time0), 'Filter on SEGMENT: ', str(OPTIMISATION_SEGMENT), ' users')
    print(get_time(settings.time0), 'Total clients in segment: ', str(len(df_ref[(df_ref['subscription_quantity'] >= OPTIMISATION_SEGMENT[0])
                                                                           & (df_ref['subscription_quantity'] <= OPTIMISATION_SEGMENT[1])])))
    print(get_time(settings.time0), 'Total ARR in segment: ', str(df_ref[(df_ref['subscription_quantity'] >= OPTIMISATION_SEGMENT[0])
                                                                   & (df_ref['subscription_quantity'] <= OPTIMISATION_SEGMENT[1])].arr.sum()))
    print(get_time(settings.time0), 'Max roadmap items:', ROADMAP_LENGTH)
    print(get_time(settings.time0), 'Fixed roadmap start:', ROADMAP_START,'\n')

    ###
    ### DATA PREPARATION
    ###
    df = df_ref[(df_ref['subscription_quantity'] >= OPTIMISATION_SEGMENT[0]) & (df_ref['subscription_quantity'] <= OPTIMISATION_SEGMENT[1])]
    df_apps = analyse_apps(df)

    for SCHEME in SCHEMES:#[0,1]
        if SCHEME == 0:
            print('Finding roadmap for optimal client growth...')
        elif SCHEME == 1:
            print('Finding roadmap for optimal revenue growth...')
        ###
        ### PREPARING ANALYSIS
        ###
        print('\n', get_time(settings.time0), 'PREPARING ANALYSIS:', '\n')
        df_apps_reduced, apps_removed = reduce_roadmap(df_apps, df_ref, ROADMAP_LENGTH, SCHEME)
        df_reduced = remove_clients(apps_removed, df)

        df_reduced_ref = remove_clients(apps_removed, df_ref) #for after-calculation

        ###
        ### EXECUTE ANALYSIS
        ###
        print('\n', get_time(settings.time0), 'EXECUTE ANALYSIS:', '\n')
        df_clientapps = df_reduced[df_apps_reduced.unique_apps]
        roadmaps = generate_roadmaps(df_apps_reduced, ROADMAP_START)
        df_roadmaps = create_roadmap_matrices(df_apps_reduced, roadmaps)
        df_roadmaps = calc_roadmap_effect(df_clientapps, df_reduced, df_roadmaps)

        df_clientapps_ref = df_reduced_ref[df_apps_reduced.unique_apps]#for after-calculation

        ###
        ### INTERPRETATION
        ###
        print('\n','INTERPRETATION:','\n')
        df_roadmap_winners = determine_roadmap_winners(df_roadmaps, SCHEME)
        dfs_results = get_resultsv2(df_clientapps_ref, df_reduced_ref, df_roadmap_winners, SEGMENTS)
        plot_results(dfs_results, df_apps_reduced, df_ref, SCHEME, ROADMAP_LENGTH, SEGMENTS, OPTIMISATION_SEGMENT)

        # dfs_results = dict()
        # roadmap_winners = determine_roadmap_winners(df_roadmaps, SCHEME)#todo: changed function to output entire df; remove old code when successful
        # for segment in SEGMENTS:
        #     print('\n','SEGMENT: ', str(segment))
        #     df_reduced_segment = remove_clients(df_apps_reduced, df_ref, segment[0], segment[1])
        #     df_results = get_results(df_roadmaps, df_reduced_segment, roadmap_winners)
        #     dfs_results[str(segment)] = df_results

        print('\n', get_time(settings.time0), 'Total duration of main: ', get_time(settings.time00), '\n')

    return None

if __name__ == "__main__":
    df_ref = data_prep(DATA)
    for OPTIMISATION_SEGMENT in OPTIMISATION_SEGMENTS:
        main(df_ref, OPTIMISATION_SEGMENT, SCHEMES)
