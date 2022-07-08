import pandas as pd
from utils import *
import itertools
from matplotlib import pyplot as plt

def data_prep(DATA):
    """
    Convert custom data file with customer data from Excel to dataframes
    """
    # read file
    print(get_time(settings.time0), 'DATAFILE:', '\n')
    data = pd.read_excel(DATA, sheet_name=0, header=0)
    df_ref = pd.DataFrame(data) # make proper dataframe

    # improve dataframe
    print(get_time(settings.time0), 'Name:', DATA)
    print(get_time(settings.time0), 'Total clients:', len(df_ref))
    print(get_time(settings.time0), 'Total ARR (€):', round(df_ref.arr.sum(),2))
    return df_ref

# def get_unique_apps(df):
#     """
#     get unique Apps as sorted list
#     """
#     unique_apps = df['Apps'].explode().sort_values().unique()
#     unique_apps = list(filter(None, unique_apps))
#     print(get_time(settings.time0), 'apps used by clients:', '\n', unique_apps)
#     return unique_apps

def analyse_apps(df):
    """
    For all unique apps save how many clients use it and the related arr
    """
    apps = [col for col in df if col.startswith('App.')]
    print(get_time(settings.time0), 'apps used by clients:', '\n', apps)

    # count use per app and sum arr per app
    df_apps = pd.DataFrame(apps, columns=['unique_apps'])
    app_counter = list()
    app_arrs = list()
    for x in df_apps['unique_apps']:
        app_count = df[x].sum()
        app_counter.append(app_count)
        app_arr = (df[x]*df.arr).sum()
        app_arrs.append(app_arr)
    df_apps['app_counter'] = app_counter
    df_apps['app_arr'] = app_arrs

    return df_apps

def reduce_roadmap(df_apps, df, ROADMAP_LENGTH, SCHEME):
    """
    reduce apps in roadmap by removing the least used/revenue generating apps until max_items remain in roadmap
    """
    if ROADMAP_LENGTH < len(df_apps):
        if SCHEME == 0:
            removed_apps = df_apps.nsmallest(len(df_apps)-ROADMAP_LENGTH, columns='app_counter')
        elif SCHEME == 1:
            removed_apps = df_apps.nsmallest(len(df_apps)-ROADMAP_LENGTH, columns='app_arr')
        df_apps_reduced = df_apps.drop(index=removed_apps.index)

        print(get_time(settings.time0), 'reduced the number apps to improve performance with:', len(removed_apps),'\n',
              'removed apps: ', '\n', removed_apps['unique_apps'].tolist(), '\n',
              'remaining apps: ', '\n', df_apps['unique_apps'].tolist())

        # if value in any removed App column is one, then remove client
        df_removed_clients = df[df[removed_apps['unique_apps']].any(1)]

        print(get_time(settings.time0), 'Associated clients due to the removal: ',len(df_removed_clients),
              '(', round(len(df_removed_clients)/len(df)*100), '%), representing €', int(df_removed_clients.arr.sum()),
              '(', round(df_removed_clients.arr.sum()/df.arr.sum()*100), '%)')

    return df_apps_reduced, removed_apps

def remove_clients(removed_apps, df, segment_start=0, segment_end=1000):
    """
    remove all clients from analysis which use an app not included in the roadmap and save new df
    """
    # df = df[(df['subscription_quantity'] >= segment_start) & (df['subscription_quantity'] <= segment_end)]
    df_reduced = df[~df[removed_apps['unique_apps']].any(1)] #remove clients using dropped apps and save new df
    df_reduced = df_reduced.drop(columns=removed_apps.unique_apps)

    return df_reduced

def generate_roadmaps(df_apps, ROADMAP_START):
    """
    generate roadmaps
    """
    print(get_time(settings.time0), 'enforce the roadmap will always start with: ', ROADMAP_START)
    df_apps_excl_roadmap_start = df_apps[~df_apps.unique_apps.isin(ROADMAP_START)] #remove apps from roadmap_start

    print(get_time(settings.time0), 'start generating roadmaps: ', np.math.factorial(len(df_apps_excl_roadmap_start['unique_apps'])))
    roadmaps_excl_roadmap_start = list(itertools.permutations(df_apps_excl_roadmap_start['unique_apps']))

    #extend roadmaps with fixed apps (use loop with extend function)
    roadmaps = []
    for r in roadmaps_excl_roadmap_start:
        roadmap = ROADMAP_START[:]
        roadmap.extend(r)
        roadmaps.append(roadmap)

    print(get_time(settings.time0), 'finished generating roadmaps: ', len(roadmaps))

    return roadmaps

def create_roadmap_matrices(df_apps, roadmaps):
    """
    create matrix representing roadmaps
    """
    df_roadmaps = pd.DataFrame(columns=df_apps['unique_apps'])
    df_roadmaps = df_roadmaps.rename_axis("apps", axis="columns")
    for c in df_roadmaps.columns:
        lst = [r.index(c) for r in roadmaps]
        # TODO: double nested for loop inline: e.g. [print(x, y) for x in iter1 for y in iter2]
        df_roadmaps[c] = lst
    print(get_time(settings.time0), 'finished roadmap matrix representations...')
    df_roadmaps['roadmaps'] = roadmaps # add column containing roadmaps

    return df_roadmaps

def calc_roadmap_effect(df_clientapps, df_reduced, df_roadmaps):
    """
    INTEGRATE CLIENTS/ARR OVER ROADMAP by CROSS MULTIPLICATION
    """
    print(get_time(settings.time0), 'start analysing client growth per roadmap...')
    client_apps = df_clientapps.to_numpy()
    client_arr = df_reduced.arr.to_numpy()
    cumsum_count = list()
    cumsum_arr = list()
    for i, r in enumerate(df_roadmaps.drop(columns='roadmaps').to_numpy()): # loop over every roadmap
        multi = r*client_apps # per client (row) filter out the non-used apps within the roadmap
        multimax = np.amax(multi, axis=1) # determine per client: the position in the roadmap which completes all apps for that client
        active_clients_count = len(r)-multimax # the number of positions in the roadmap where the needs of the clients are fullfilled
        active_clients_countsum = sum(active_clients_count) # Sum over all clients: the number of positions in the roadmap where the needs of the clients are fullfilled
        active_clients_arrsum = sum(active_clients_count*client_arr)
        # df_roadmaps.loc[[i], 'cumsum_count'] = active_clients_countsum
        cumsum_count.append(active_clients_countsum)
        # df_roadmaps.loc[[i], 'cumsum_arr'] = active_clients_arrsum
        cumsum_arr.append(active_clients_arrsum)
        if ((i+1) % round(len(df_roadmaps)/20)) == 0:
            print(get_time(settings.time0), 'i =', i, round(i/len(df_roadmaps)*100,0), '%')#end='\r'
    df_roadmaps['cumsum_count'] = cumsum_count
    df_roadmaps['cumsum_arr'] = cumsum_arr
    # print(get_time(settings.time0))

    return df_roadmaps

def determine_roadmap_winners(df_roadmaps, SCHEME):
    if SCHEME == 0: # 0 = optimisation based on client growth,
        # roadmap_winners = [df_roadmaps['roadmaps'][i] for i in np.argwhere(df_roadmaps['cumsum_count'].to_numpy() == np.amax(df_roadmaps['cumsum_count'].to_numpy())).flatten().tolist()]
        roadmap_winners = df_roadmaps[df_roadmaps['cumsum_count'] == df_roadmaps['cumsum_count'].max()]
        # print('Max client sum per roadmap for target group: ', df_roadmaps['cumsum_count'].max()) #to verify calc
    elif SCHEME == 1: # 1 = optimisation based on arr growth
        # roadmap_winners = [df_roadmaps['roadmaps'][i] for i in np.argwhere(df_roadmaps['cumsum_arr'].to_numpy() == np.amax(df_roadmaps['cumsum_arr'].to_numpy())).flatten().tolist()]
        roadmap_winners = df_roadmaps[df_roadmaps['cumsum_arr'] == df_roadmaps['cumsum_arr'].max()]
        # print('Max client arr per roadmap for target group: €', round(df_roadmaps['cumsum_arr'].max(),2)) #to verify calc
    print('Corresponding roadmap(s):')
    print(*roadmap_winners.roadmaps, sep='\n')
    # print(*roadmap_winners, sep='\n')

    return roadmap_winners

def get_results(df_roadmaps, df, roadmap_winners):
    """
    Determines client growth and revenue growth for a given roadmap(s) and combines the results in df_results
    """
    # Create a 3 dimensional matrix, where for each roadmap it is determined when a client has all Apps he needs
    clients_per_roadmap = list()
    client_cumsum = list()
    arr_per_roadmap = list()
    arr_cumsum = list()
    df_apps_set = df.Apps.to_numpy()
    df_arr = df.arr.to_numpy()
    for roadmap in roadmap_winners: #(loop over roadmaps)
        client_roadmap = np.zeros((len(df), len(roadmap)))
        arr_roadmap = np.zeros((len(df), len(roadmap)))
        for i in range(len(df_roadmaps['roadmaps'])):
            roadmap_iter = set(roadmap[:i+1]) # partial roadmap representing roadmap progress so far
            for c in range(len(df)):
                match = set(df_apps_set[c]).issubset(roadmap_iter) #todo: optimise performance, but how?
                if match: # are all client Apps available?
                    client_roadmap[c, i:] = 1
                    arr_roadmap[c, i:] = df_arr[c]

        #Alternative, more lean, approach:
        #client_count = np.sum(client_roadmap) + sum(df['Apps'].isna()) #client growth per roadmap
        #clients_per_roadmap.append(client_count)
        client_count = np.sum(client_roadmap, axis=0) #client growth per roadmap
        clients_per_roadmap.append(client_count)
        client_cumsum.append(np.sum(client_count))
        arr = np.sum(arr_roadmap, axis=0)
        arr_per_roadmap.append(arr)
        arr_cumsum.append(np.sum(arr))
        # if (r % 100) == 0:
        #     print(time.time() - time0, 'b', r); time0=time.time()

    df_results = pd.DataFrame([clients_per_roadmap], index=['clients']).T
    df_results['clients_cumsum'] = client_cumsum
    df_results['arr'] = arr_per_roadmap
    df_results['arr_cumsum'] = arr_cumsum
    df_results['roadmap'] = roadmap_winners

    # pd.set_option('display.max_columns', ROADMAP_LENGTH)
    print('Max cumsum clients for current segment: ','\n',df_results['clients_cumsum'].max())
    print('Max cumsum arr for current segment: €','\n', round(df_results['arr_cumsum'].max(),2))
    print('Client growth: ', *df_results['clients'].tolist(), sep='\n')
    print('ARR growth: ', *df_results['arr'].tolist(), sep='\n')

    return df_results

def get_resultsv2(df_clientapps, df, df_roadmap_winners, SEGMENTS):
    """
    INTEGRATE CLIENTS/ARR OVER ROADMAP by CROSS MULTIPLICATION
    """
    print(get_time(settings.time0), 'start analysing client growth per roadmap...')

    client_apps = df_clientapps.to_numpy()
    client_arr = df.arr.to_numpy()

    segment_vectors = list()
    for segment in SEGMENTS:
        segment_vector = (df['subscription_quantity'] >= segment[0]) & (df['subscription_quantity'] <= segment[1])
        segment_vector = np.multiply(segment_vector,1)
        segment_vectors.append(segment_vector)

    dfs_results = dict()
    clients_per_roadmap = [ [] for _ in range(len(SEGMENTS)) ]
    arr_per_roadmap = [ [] for _ in range(len(SEGMENTS)) ]
    client_cumsum = [0] * len(SEGMENTS)
    arr_cumsum = [0] * len(SEGMENTS)
    for i, r in enumerate(df_roadmap_winners.drop(columns=['roadmaps','cumsum_count','cumsum_arr']).to_numpy()): # loop over every roadmap
        multi = r*client_apps # per client (row) filter out the non-used apps within the roadmap
        multimax = np.amax(multi, axis=1) # determine per client: the position in the roadmap which completes all apps for that client
        client_count_matrix = np.zeros(client_apps.shape)
        for c in range(len(client_count_matrix)):
            client_count_matrix[c, multimax[c]:] = 1 #fill matrix with ones where client is fully served
        arr_matrix = client_count_matrix * client_arr[:, np.newaxis] #fill matrix with arr where client is fully served

        for s in range(len(SEGMENTS)):
            client_count_matrix_segment = client_count_matrix * segment_vectors[s][:, np.newaxis]
            arr_matrix_segment = arr_matrix * segment_vectors[s][:, np.newaxis]

            client_count = sum(client_count_matrix_segment)
            clients_per_roadmap[s].append(client_count)
            arr = sum(arr_matrix_segment)
            arr_per_roadmap[s].append(arr)
            client_cumsum[s] = sum(client_count)
            arr_cumsum[s] = sum(arr)

    for s, segment in enumerate(SEGMENTS):
        print('\n','SEGMENT: ', str(segment))

        df_results = pd.DataFrame([clients_per_roadmap[s]], index=['clients']).T
        df_results['clients_cumsum'] = client_cumsum[s]
        df_results['arr'] = arr_per_roadmap[s]
        df_results['arr_cumsum'] = arr_cumsum[s]
        df_results['roadmap'] = df_roadmap_winners.roadmaps.tolist()

        print('Max cumsum clients for current segment: ','\n',df_results['clients_cumsum'].max())
        print('Max cumsum arr for current segment: €','\n', round(df_results['arr_cumsum'].max(),2))
        print('Client growth: ', *df_results['clients'].tolist(), sep='\n')
        print('ARR growth: ', *df_results['arr'].tolist(), sep='\n')

        dfs_results[str(segment)] = df_results

    return dfs_results

def plot_results(dfs, df_appuse, df_ref, SCHEME, ROADMAP_LENGTH, SEGMENTS, OPTIMISATION_SEGMENT):
    fig, axs = plt.subplots(2,2)

    #reduce dataframe to one solution if multiple solutions exist
    for segment in SEGMENTS:
        dfs[str(segment)] = dfs[str(segment)][dfs[str(segment)].index == 0]

    #create a stacked line plot for # clients
    df_clients = pd.DataFrame()
    for segment in SEGMENTS:
        df_clients[str(segment)] = dfs[str(segment)].clients.explode().reset_index(drop=True)
    # axs[0, 0].plot(pd.DataFrame(df_results['clients'].tolist(), index= df_results.index).T) #line plot
    axs[0, 0].plot([0,len(df_appuse)-1], [len(df_ref), len(df_ref)], label='All clients')
    axs[0, 0].set_title('# clients')
    df_clients.plot.area(ax=axs[0, 0])

    #create a stacked line plot for ARR
    df_arr = pd.DataFrame()
    for segment in SEGMENTS:
        df_arr[str(segment)] = dfs[str(segment)].arr.explode().reset_index(drop=True)
    axs[0, 1].plot([0,len(df_appuse)-1], [df_ref.arr.sum(), df_ref.arr.sum()], label='ARR_all clients')
    axs[0, 1].set_title('ARR')
    df_arr.plot.area(ax=axs[0, 1])
    #axs[0, 1].plot(pd.DataFrame(df_results['arr'].tolist(), index= df_results.index).T)  #line plot

    #create a line plot for all segments with % clients wrt total clients in segment
    totals_clients = list()
    for segment in SEGMENTS:
        total_clients = len(df_ref[(df_ref['subscription_quantity'] >= segment[0]) & (df_ref['subscription_quantity'] <= segment[1])])
        totals_clients.append(total_clients)
    print('Cumsum clients of all segments: €','\n', sum(totals_clients))
    # axs[1, 0].plot(pd.DataFrame((df_results['clients']/len(df_ref)).tolist(), index= df_results.index).T, label='client')
    axs[1, 0].set_title('Client (%)')
    axs[1, 0].set_ylim([0, 1.0])
    df_clients.divide(totals_clients).plot(ax=axs[1,0])

    #create a line plot for all segments with % arr wrt total arr in segment
    totals_arr = list()
    for segment in SEGMENTS:
        total_arr = df_ref[(df_ref['subscription_quantity'] >= segment[0]) & (df_ref['subscription_quantity'] <= segment[1])].arr.sum()
        totals_arr.append(total_arr)
    print('Cumsum arr of all segments: €','\n', sum(totals_arr))

    # axs[1, 1].plot(pd.DataFrame((df_results['arr']/df_ref.arr.sum()).tolist(), index= df_results.index).T, label='arr')
    axs[1, 1].set_title('ARR (%)')
    axs[1, 1].set_ylim([0, 1.0])
    df_arr.divide(totals_arr).plot(ax=axs[1,1])
    # plt.show()
    roadmap_as_title = dfs[str(SEGMENTS[0])].roadmap[0]
    if ROADMAP_LENGTH > 7:
        roadmap_as_title = str(roadmap_as_title[0:len(roadmap_as_title)//2])[:-2] + '\n' \
                           + str(roadmap_as_title[len(roadmap_as_title)//2:])[1:]
    else:
        roadmap_as_title = str(roadmap_as_title)

    if SCHEME == 0:
        fig.suptitle('Growth vs roadmap (' + 'optimised by client growth for ' + str(OPTIMISATION_SEGMENT) + ' users)' + '\n' +
                     'Cum(ARR) = €' + str(int(df_arr.sum().sum())) + ' | Cum(clients) = ' + str(int(df_clients.sum().sum())) + '\n'
                     + roadmap_as_title)
    elif SCHEME == 1:
        fig.suptitle('Growth vs roadmap ('+ 'optimised by revenue for ' + str(OPTIMISATION_SEGMENT) + ' users)' + '\n' +
                     'Cum(ARR) = €' + str(int(df_arr.sum().sum())) + ' | Cum(clients) = ' + str(int(df_clients.sum().sum())) + '\n'
                     + roadmap_as_title)

    fig.set_size_inches(8, 10)
    fig.savefig('Results/SCHEME_'+str(SCHEME)+'-USERS_'+str(OPTIMISATION_SEGMENT)+'-ROADMAP_LENGTH_'+str(ROADMAP_LENGTH)+'.png', dpi=80)
    plt.close(fig)
    time.sleep(2)

    return None