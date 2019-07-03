"""Tests.
!!!

"""


#from tcpo import get_args
#import doctest
#
#
#argv = ['--action ', 'extract&classify']
#try:
#    get_args(argv)
#except Exception as ex:
#    print(ex)
"""
    #args = parser.parse_args(['--datasets', 'MondialMarmi20', '--descriptor', 'RankTransform'])
    #args = parser.parse_args(['--datasets', 'NewBarkTex', '--descriptor', 'CompletedLocalBinaryCountSM', '--order', 'linear', '--radius', '1', '--action', 'all'])
    #args = parser.parse_args(['--datasets', 'CBT', '--descriptor', 'LocalConcaveConvexMicroStructurePatterns', '--order', 'random', '--radius', '1', '--action', 'all'])
    #args = parser.parse_args(['--datasets', 'NewBarkTex', '--order', 'random', '--action', 'features'])
    #args = parser.parse_args('--jobname kk --action extract&classify'.split())
    args = parser.parse_args('--descriptor CompletedLocalBinaryCountSM RankTransform --action ef --dataset NewBarkTex CBT'.split())
    #args = parser.parse_args(['--jobname', 'default', '--action', 'latex'])
    #args = parser.parse_args(['--jobname', 'tutti', '--action', 'job'])

    return args



def extract_features(data, loops):
    print('Extracting features...')

def classify(data, loops, test_size, n_tests):
    print('Classifying...')

def job_script(data, loops, job_id, partition):
    print('Generating job script...')

def generate_latex(args):
    print('Generating LaTeX...')

def delete_features(folder, loops):
    print('Deleting features...')

def delete_results(data, loops):
    print('Deleting results...')


def main():
    pass

#%%===========
# MAIN PROGRAM
#=============
if __name__ == '__main__':
    
    # Print welcome message
    utils.boxed_text('TEXTURE CLASSIFICATION')

    # Initialize global variables
    try:
        import config
    except ImportError:
        print('config.py not found')
        print('Using default settings')
        config = Configuration()
    
    # Parse command-line options and arguments
    args = get_args(sys.argv)
    
    # Abort execution if datasets or descriptors are not passed in
    if (args.dataset is None) or (args.descriptor is None):
        sys.exit('error: datasets or descriptors have not been passed in')
        
#    # Make sure data directory exists
#    IPython.utils.path.ensure_dir_exists(config.DATA)
#
#    # Set up the lists of datasets and descriptors
    feature_loops = None#load_settings(arguments, config.IMGS)
    results_loops = None#feature_loops + (config.ESTIMATORS,)
#
    # Execute the proper action
    option = args.action.lower()
    if option in ('extract_features', 'ef'):
        extract_features(config.data, feature_loops)
    elif option in ('classify', 'c'):
        classify(config.data, results_loops, config.test_size, config.n_tests)
    elif option in ('extract&classify', 'ec'):
        extract_features(config.data, feature_loops)
        classify(config.data, results_loops, config.test_size, config.n_tests)
    elif option in ('job', 'j'):
        job_script(config.data, results_loops, args.jobname, args.partition)
    elif option in ('latex', 'l'):
        generate_latex(args)
    elif option in ('delete_features', 'df'):
        delete_features(config.data, feature_loops)
    elif option in ('delete_results', 'dr'):
        delete_results(config.data, results_loops)
    elif option in ('delete_all', 'da'):
        delete_features(config.data, feature_loops)
        delete_results(config.data, results_loops)
    else:
        print(f'Action {args.action} is not valid\n')
        print('Run:')
        print('    $ python tcpo.py --help')
        print('for help on command-line options and arguments')

    print(args)


#print('Mean test score: {}'.format(gscv.cv_results_['mean_test_score']))
#print('Best score : {}'.format(gscv.best_score_))
#print('Best params : {}'.format(gscv.best_params_))
#print('Score on test : {}\n\n'.format(test_score))

#print('parser.args = {}\n'.format(args))
#print('params = {}\n'.format(params))
#print('datasets = {}'.format(datasets))
#print('sys.argv = {}\n'.format(sys.argv))
#img = io.imread(r'C:\texture\images\MondialMarmi20\00\AcquaMarina_A_00_01.bmp')

#if not os.path.exists(log):
#    os.makedirs(log)
#log_file = os.path.join(log, 'log ' + datetime.datetime.now().__str__().split('.')[0].replace(':', '.') + '.txt')
#log = open(log_file, 'w')
"""